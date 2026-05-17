"""LangGraph ReAct agent — port of fashion-recommend/ai/agent.go.

Key design decisions vs. Go:
  * StateGraph(AgentState) replaces the hand-rolled for-loop in AgentChat().
  * ToolNode replaces executeToolCall() + injectDefaultUserID() + the
    manual role="tool" message appending (~40 lines of Go).
  * Model tiering — two separate models:
      router_model  (gemini-2.5-flash) — function-calling capable; makes every
                    tool decision across all ReAct iterations.
      final_model   (gemma-3-27b-it)  — text-generation only; writes the one
                    polished answer at the end.
    Gemma does not support function calling, so finalizer_node reformats the
    full message history (which contains ToolMessages) into a plain
    HumanMessage before invoking it.
  * user_id lives in AgentState and is injected into tools via InjectedState.
  * trace is accumulated in AgentState across router_node iterations.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent.tools import make_tools
from db.client import DBClient
from db.gorse_client import GorseClient
from traits.gorse_sync import GorseSync

# ---------------------------------------------------------------------------
# System prompt — matches Go's buildInitialMessages() verbatim
# ---------------------------------------------------------------------------
# 原中文 prompt（保留供参考）:
# _SYSTEM_PROMPT_TEMPLATE = """你是「时尚小助手」，一个专业的时尚品牌推荐 AI 代理。
# 当前用户ID：{user_id}
#
# 你拥有以下工具，在回答用户问题前请合理使用它们：
# - get_recommendations：获取个性化商品推荐或相似商品
# - get_user_preferences：获取当前用户已保存的时尚偏好（风格、颜色、价格、品牌等），无需传参
# - get_item_details：获取指定商品的名称、品类和属性标签
# - search_fashion_trends：搜索当前时尚趋势和流行资讯
# - update_user_traits：当用户在对话中明确表达了风格偏好（如「我喜欢简约风」）、颜色偏好、价格敏感度或品牌偏好时，调用此工具暂存偏好更新，等待用户确认后写入系统。每次对话最多调用一次。
#
# 在回答涉及推荐、偏好或购物的问题时，优先调用 get_user_preferences 了解用户品味，再调用 get_recommendations 获取商品列表。
# 始终以友好、专业的语气用中文回答用户。"""
_SYSTEM_PROMPT_TEMPLATE = """You are "Fashion Curator", a professional AI agent for a fashion brand recommendation system.
Current user ID: {user_id}

You have the following tools available. Use them appropriately before answering the user's question:
- get_recommendations: Fetch personalised product recommendations or find items similar to a given product.
- get_user_preferences: Retrieve the current user's saved fashion preferences (style, colour, price, brands, etc.). No parameters needed.
- get_item_details: Get the name, category, and attribute labels for a specific product.
- search_fashion_trends: Search for current fashion trends, seasonal styles, and brand news.
- update_user_traits: When the user explicitly states a style preference (e.g. "I love minimalist style"), colour preference, price sensitivity, or brand preference during the conversation, call this tool to stage the update for user confirmation before writing to the system. Call at most once per conversation turn.

When answering questions about recommendations, preferences, or shopping, prioritise calling get_user_preferences first to understand the user's taste, then call get_recommendations to fetch products.
Always respond in a friendly and professional tone in English."""


# ---------------------------------------------------------------------------
# Public data models (returned to the handler)
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    iteration: int
    thought: str
    action: str
    action_input: str
    observation: str


class AgentResult(BaseModel):
    answer: str
    trace: list[TraceStep]
    iterations: int
    tokens_used: int = 0
    pending_approval: bool = False
    pending_trait_updates: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    trace: list[dict]
    iterations: int
    tokens_used: int
    pending_trait_updates: list[dict]  # staged by update_user_traits tool, flushed by write_traits node


# ---------------------------------------------------------------------------
# Quality gate helper
# ---------------------------------------------------------------------------

def _has_signal(content: str) -> bool:
    """Return True if a tool result contains actionable data (not empty/error)."""
    content = (content or "").strip()
    if not content or content in ("[]", "{}", "null"):
        return False
    # 原中文 markers（保留供参考，对应 tools.py 旧版本返回的错误字符串）:
    # if any(marker in content for marker in ["商品搜索失败", "未找到商品"]):
    if any(marker in content for marker in ["Product search failed", "Item not found", "商品搜索失败", "未找到商品"]):
        return False
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return len(parsed) > 0
        if isinstance(parsed, dict):
            return parsed.get("status") != "no_data"
        return parsed is not None
    except (json.JSONDecodeError, ValueError):
        return len(content) > 30  # non-JSON (e.g. Tavily results) need some body


# ---------------------------------------------------------------------------
# Trait merge helper
# ---------------------------------------------------------------------------

def _merge_trait_updates(existing: dict, updates: list[dict]) -> dict:
    """Merge a list of staged trait update dicts into an existing traits dict.

    Rules (mirror the extractor's _merge logic):
      - style_preferences / color_preferences: update scores (staged value wins).
      - price_sensitivity: last non-empty value wins.
      - brand_preferences / occasions / keywords / interests: union, deduplicated.
    """
    # Start from a deep-ish copy of existing so we don't mutate the DB row.
    merged: dict = {
        "style_preferences": dict(existing.get("style_preferences") or {}),
        "color_preferences": dict(existing.get("color_preferences") or {}),
        "price_sensitivity": existing.get("price_sensitivity") or "",
        "brand_preferences": list(existing.get("brand_preferences") or []),
        "occasions": list(existing.get("occasions") or []),
        "keywords": list(existing.get("keywords") or []),
        "interests": list(existing.get("interests") or []),
    }

    for upd in updates:
        if upd.get("style_preferences"):
            merged["style_preferences"].update(upd["style_preferences"])
        if upd.get("color_preferences"):
            merged["color_preferences"].update(upd["color_preferences"])
        if upd.get("price_sensitivity"):
            merged["price_sensitivity"] = upd["price_sensitivity"]
        for field in ("brand_preferences", "occasions", "keywords", "interests"):
            if upd.get(field):
                seen = set(merged[field])
                for item in upd[field]:
                    if item and item not in seen:
                        merged[field].append(item)
                        seen.add(item)

    return merged


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    api_key: str
    # Router: must support function calling — only Gemini models qualify.
    # Finalizer: text-generation only, no tool calls needed — Gemma works fine.
    router_model: str = "gemini-2.5-flash"  # Gemini: function-calling capable
    final_model: str = "gemma-3-27b-it"     # Gemma: strong writer, free tier
    max_iterations: int = 8
    token_budget: int = 20_000  # exit ReAct loop early if cumulative tokens exceed this


# ---------------------------------------------------------------------------
# AgentGraph
# ---------------------------------------------------------------------------

class AgentGraph:
    """Wraps the compiled LangGraph graph and exposes a single chat() method."""

    def __init__(
        self,
        config: AgentConfig,
        db: DBClient,
        gorse: GorseClient,
        checkpointer=None,
    ) -> None:
        self._config = config
        self._checkpointer = checkpointer
        self._db = db
        self._gorse_sync = GorseSync(db, gorse)
        self._tools = make_tools(db, gorse)

        # Router: Gemini with tools bound — makes every tool-call decision.
        self._router_model = ChatGoogleGenerativeAI(
            model=config.router_model,
            google_api_key=config.api_key,
        ).bind_tools(self._tools)

        # Finalizer: Gemma, NO tools bound — only does text synthesis.
        # Called exactly once; receives a reformatted plain-text prompt
        # (no ToolMessage objects, which Gemma's API doesn't support).
        self._final_model = ChatGoogleGenerativeAI(
            model=config.final_model,
            google_api_key=config.api_key,
        )

        self._compiled = self._build_graph()

    def _build_graph(self):
        max_iter = self._config.max_iterations
        token_budget = self._config.token_budget

        # ---- router_node ----
        # Mirrors the per-iteration body of Go's AgentChat() for-loop.
        async def router_node(state: AgentState) -> dict:

            # thinking: The AI examines the history and determines whether to use a tool or answer the user directly
            # ainvoke stands for Asynchronous Invoke of the model
            # After the AI responds, resp will typically contain either:A content string if the AI has the answer. OR Tool calls if the AI needs more information.
            resp: AIMessage = await self._router_model.ainvoke(state["messages"])

            # Tracking Loops: iterations counter is incremented.--> prevent the agent from looping indefinitely if it encounters an issue.
            new_iter = state.get("iterations", 0) + 1

            # Trace Logging: A "snapshot" of the agent's internal thought process is created.
            # used for debugging and the "Thinking..." UI in modern AI chat apps.
            trace = list(state.get("trace") or [])

            # safely returns None if the AI didn't request a tool.
            tool_calls = getattr(resp, "tool_calls", None)
            if tool_calls:
                # Record the first tool call as a TraceStep (observation filled
                # by ToolNode; we leave it blank here as Go does before execution).
                # gemini-2.5-flash returns content as a list of blocks when thinking
                # is enabled — coerce to str for TraceStep.thought.
                tc = tool_calls[0]
                thought = resp.content
                if isinstance(thought, list):
                    thought = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in thought
                    ).strip()
                trace.append(
                    {
                        "iteration": new_iter,
                        "thought": thought or "",
                        "action": tc["name"],
                        "action_input": str(tc.get("args", {})),
                        "observation": "",
                    }
                )

            # Accumulate token usage — usage_metadata is ephemeral on the response
            # object and cannot be reconstructed from message history (Fix 2A pattern).
            usage = getattr(resp, "usage_metadata", None) or {}
            new_tokens = state.get("tokens_used", 0) + usage.get("total_tokens", 0)

            return {"messages": [resp], "iterations": new_iter, "trace": trace, "tokens_used": new_tokens}

        # ---- finalizer_node ----
        # Calls Gemma (no function-calling support) to write the polished answer.
        # Gemma's API rejects ToolMessage objects, so we reformat the full history
        # into a single HumanMessage that lists the user question + tool outputs.
        async def finalizer_node(state: AgentState) -> dict:
            messages = state["messages"]

            # Scope to the current turn only — same reasoning as quality_gate_route:
            # ToolMessages from prior turns must not bleed into this turn's answer.
            last_human_idx = max(
                (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
                default=-1,
            )

            # --- collect user question and current-turn tool observations ---
            user_question = ""
            tool_observations: list[str] = []
            # True when get_recommendations ran this turn but returned no products.
            # Used below to add an explicit "don't invent product names" guard so
            # Gemma doesn't confabulate items when only preference data is present.
            rec_called_empty = False
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage) and not isinstance(msg, ToolMessage):
                    # Keep the last HumanMessage as the user's question.
                    content = msg.content
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        ).strip()
                    user_question = content
                elif isinstance(msg, ToolMessage) and i > last_human_idx:
                    name = getattr(msg, "name", "tool")
                    tool_observations.append(f"【{name}】\n{msg.content}")
                    if name == "get_recommendations" and not _has_signal(msg.content):
                        rec_called_empty = True

            # --- build Gemma-compatible prompt ---
            if tool_observations:
                gathered = "\n\n".join(tool_observations)
                # 原中文 prompt（保留供参考）:
                # no_products_note = (
                #     "\n重要：商品搜索结果为空，请不要编造或推测任何具体商品名称。"
                #     "请根据用户偏好提供风格建议，并说明暂时没有找到匹配商品。"
                #     if rec_called_empty else ""
                # )
                no_products_note = (
                    "\nIMPORTANT: The product search returned no results. Do NOT invent or guess specific product names. "
                    "Instead, offer style advice based on the user's preferences and explain that no matching products were found right now."
                    if rec_called_empty else ""
                )
                # 原中文 prompt（保留供参考）:
                # prompt = (
                #     f"用户问题：{user_question}\n\n"
                #     f"以下是为了回答该问题已收集到的信息：\n\n{gathered}\n\n"
                #     f"请根据以上信息，以友好、专业的语气用中文给出完整的回答。"
                #     f"如果信息中包含具体商品，请直接引用商品名称（name字段）进行介绍，不要使用模糊表述如「一些商品」或「相关商品」。"
                #     + no_products_note
                # )
                prompt = (
                    f"User question: {user_question}\n\n"
                    f"Here is the information collected to answer this question:\n\n{gathered}\n\n"
                    f"Based on the above information, please provide a complete and helpful answer in a friendly and professional tone in English. "
                    f"If the information includes specific products, refer to them by their actual name (the 'name' field) — do not use vague phrases like 'some products' or 'related items'."
                    + no_products_note
                )
            else:
                # No tools were called — answer from general knowledge.
                # Nudge fires on both exit conditions: iteration cap and token budget.
                exhausted = (
                    state.get("iterations", 0) >= max_iter
                    or state.get("tokens_used", 0) >= token_budget
                )
                # 原中文 prompt（保留供参考）:
                # prompt = (
                #     f"用户问题：{user_question}\n\n"
                #     f"请以友好、专业的语气用中文给出完整的回答。"
                #     + ("\n\n（注：已达到最大思考轮次，请基于现有信息作答。）" if exhausted else "")
                # )
                prompt = (
                    f"User question: {user_question}\n\n"
                    f"Please provide a complete and helpful answer in a friendly and professional tone in English."
                    + ("\n\n(Note: Maximum reasoning iterations reached — please answer based on the information available.)" if exhausted else "")
                )

            resp: AIMessage = await self._final_model.ainvoke(
                [HumanMessage(content=prompt)]
            )
            # Accumulate finalizer token cost for accurate per-turn total in AgentResult.
            usage = getattr(resp, "usage_metadata", None) or {}
            new_tokens = state.get("tokens_used", 0) + usage.get("total_tokens", 0)
            return {"messages": [resp], "tokens_used": new_tokens}

        # ---- quality_gate_node ----
        # No-op waypoint: exists so LangGraph can attach a conditional edge here.
        # The actual routing logic lives in quality_gate_route below.
        async def quality_gate_node(state: AgentState) -> dict:
            return {}

        # ---- fallback_node ----
        # Deterministic honest response when all tool results are empty or errors.
        # No LLM call — prevents the finalizer from confabulating on empty data.
        async def fallback_node(state: AgentState) -> dict:
            # 原中文 fallback（保留供参考）:
            # msg = AIMessage(
            #     content=(
            #         "抱歉，我暂时没有找到与您问题相关的商品或信息。"
            #         "这可能是因为商品库中目前没有匹配的结果，或搜索工具未能获取到数据。"
            #         "您可以尝试换个描述方式，或告诉我更多具体需求，我会重新为您查找。"
            #     )
            # )
            msg = AIMessage(
                content=(
                    "I'm sorry, I couldn't find any products or information related to your question right now. "
                    "This may be because there are no matching results in our catalogue, or the search tool was unable to retrieve data. "
                    "Try rephrasing your request, or share more specific details and I'll do my best to find something for you."
                )
            )
            return {"messages": [msg]}

        # ---- routing logic ----
        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            # Branch A — iteration cap: router hit the max loop count.
            if state.get("iterations", 0) >= max_iter:
                return "quality_gate"

            # Branch B — token budget: cumulative cost exceeds the per-turn cap.
            # Fires before the tool-call check so an over-budget response doesn't
            # trigger another expensive tool + router cycle.
            if state.get("tokens_used", 0) >= token_budget:
                return "quality_gate"

            # Branch C — continue ReAct loop: router decided to call a tool.
            if getattr(last, "tool_calls", None):
                return "tools"
                
            # Branch D — clean stop: router produced a plain-text response,
            # meaning it judged no more tools are needed.
            return "quality_gate"

        def quality_gate_route(state: AgentState) -> str:
            msgs = state["messages"]
            # Scope to the current turn only: ToolMessages that appear after the
            # last HumanMessage. Without this, prior-turn ToolMessages (kept in
            # state by the add_messages reducer + checkpointer) would poison the
            # signal check — a previous successful turn would mask a current
            # turn where all tools returned empty results.
            last_human_idx = max(
                (i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)),
                default=-1,
            )
            tool_msgs = [
                m for i, m in enumerate(msgs)
                if isinstance(m, ToolMessage) and i > last_human_idx
            ]
            if not tool_msgs:
                return "finalizer"  # no tools called; router answered from general knowledge
            if any(_has_signal(m.content) for m in tool_msgs):
                return "finalizer"
            return "fallback"

        # ---- write_traits_node ----
        # Runs AFTER the finalizer/fallback produces the answer.
        # interrupt_before=["write_traits"] pauses the graph here so the
        # frontend can ask the user to approve or reject the staged updates
        # before anything is written to the DB.
        #
        # On APPROVE: resume() calls ainvoke(None, config) → this node fires.
        # On REJECT:  resume() calls aupdate_state(..., as_node="write_traits")
        #             which advances the cursor past this node to END without
        #             executing the body.
        async def write_traits_node(state: AgentState) -> dict:
            updates = state.get("pending_trait_updates") or []
            if not updates:
                return {"pending_trait_updates": []}

            user_id = state["user_id"]

            # Read current traits from DB so we merge rather than overwrite.
            existing_row = await self._db.get_user_traits(user_id)
            existing_traits: dict = (existing_row or {}).get("traits") or {}
            existing_confidence: float = (existing_row or {}).get("confidence_score", 0.5)

            merged = _merge_trait_updates(existing_traits, updates)

            # Small confidence bump when the user explicitly confirms preferences.
            new_confidence = min(existing_confidence + 0.1, 1.0)

            await self._db.save_user_traits(user_id, merged, new_confidence)
            await self._gorse_sync.sync_user_traits(user_id)

            return {"pending_trait_updates": []}

        # ---- should_write_traits routing ----
        # Only route through write_traits (and trigger the interrupt) when there
        # are actually staged updates.  If the user never expressed a preference
        # this turn, go straight to END — no approval prompt needed.
        def should_write_traits(state: AgentState) -> str:
            if state.get("pending_trait_updates"):
                return "write_traits"
            return END

        # ---- wire graph ----
        graph = StateGraph(AgentState)
        graph.add_node("router", router_node)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_node("quality_gate", quality_gate_node)
        graph.add_node("finalizer", finalizer_node)
        graph.add_node("fallback", fallback_node)
        graph.add_node("write_traits", write_traits_node)
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            should_continue,
            {"tools": "tools", "quality_gate": "quality_gate"},
        )
        graph.add_edge("tools", "router")
        graph.add_conditional_edges(
            "quality_gate",
            quality_gate_route,
            {"finalizer": "finalizer", "fallback": "fallback"},
        )
        # Both answer nodes feed into should_write_traits: only take the
        # write_traits branch when there are pending updates to approve.
        graph.add_conditional_edges(
            "finalizer",
            should_write_traits,
            {"write_traits": "write_traits", END: END},
        )
        graph.add_conditional_edges(
            "fallback",
            should_write_traits,
            {"write_traits": "write_traits", END: END},
        )
        graph.add_edge("write_traits", END)
        return graph.compile(
            checkpointer=self._checkpointer,
            # Pause BEFORE write_traits so the user can approve/reject.
            # The graph serialises to the checkpointer here; resume() or
            # reject() advances it from this exact point.
            interrupt_before=["write_traits"],
        )


    async def chat(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
    ) -> AgentResult:
        """ the only public method on AgentGraph & the single entry point the API handler calls to run the agent.
        Run the ReAct loop and return the final answer + trace.

        With checkpointing enabled, conversation history is reconstructed
        automatically from PostgreSQL — the caller only sends the new message.
        session_id maps to LangGraph's thread_id, giving each conversation
        its own isolated checkpoint.
        """
        config = {"configurable": {"thread_id": session_id}}
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(user_id=user_id)

        # Detect first turn: if no messages are checkpointed yet, seed the
        # system prompt. On subsequent turns it's already in the checkpoint and
        # adding it again would grow the context window every call.
        existing = await self._compiled.aget_state(config)
        if not existing.values.get("messages"):
            seed_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        else:
            seed_messages = [HumanMessage(content=user_message)]

        # trace, iterations, and tokens_used have no reducer (plain assignment) —
        # reset each turn so AgentResult reflects only the current request.
        # pending_trait_updates also resets: a new user message starts a fresh
        # approval cycle (the frontend blocks new messages during a pending approval).
        initial: AgentState = {
            "messages": seed_messages,
            "user_id": user_id,
            "trace": [],
            "iterations": 0,
            "tokens_used": 0,
            "pending_trait_updates": [],
        }
        result = await self._compiled.ainvoke(initial, config)

        # gemini-2.5-flash can return content as a list of content blocks
        # (e.g. thinking + text); extract plain text.
        raw_content = result["messages"][-1].content
        if isinstance(raw_content, list):
            answer = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw_content
                if not (isinstance(b, dict) and b.get("type") == "thinking")
            ).strip()
        else:
            answer = raw_content or ""
        raw_trace: list[dict] = result.get("trace") or []
        trace = [TraceStep(**t) for t in raw_trace]

        # Detect HITL interrupt: if pending_trait_updates is still non-empty after
        # ainvoke returns, the graph paused at interrupt_before=["write_traits"].
        # write_traits_node clears the list on completion, so a non-empty list here
        # means the graph is suspended and waiting for the user to approve or reject.
        staged: list[dict] = result.get("pending_trait_updates") or []

        return AgentResult(
            answer=answer,
            trace=trace,
            iterations=result.get("iterations", 0),
            tokens_used=result.get("tokens_used", 0),
            pending_approval=bool(staged),
            pending_trait_updates=staged,
        )

    async def resume(self, session_id: str, approved: bool) -> None:
        """Resume a graph that is suspended at interrupt_before=["write_traits"].

        approved=True  → run write_traits_node (DB write + Gorse sync).
        approved=False → skip write_traits by pretending it already ran with an
                         empty pending list (aupdate_state as_node advances the
                         graph cursor to END without executing the node body).
        """
        config = {"configurable": {"thread_id": session_id}}
        if approved:
            # Resume normally — write_traits_node fires and flushes the updates.
            await self._compiled.ainvoke(None, config)
        else:
            # Inject the node's "result" directly so LangGraph advances past it.
            # as_node="write_traits" tells the checkpointer that write_traits ran
            # and returned {"pending_trait_updates": []}, moving the cursor to END.
            await self._compiled.aupdate_state(
                config,
                {"pending_trait_updates": []},
                as_node="write_traits",
            )
