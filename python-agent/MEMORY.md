# python-agent — 分析笔记（Analysis Notes）

> 做成本 / 延迟分析时先读这个文件。这里记的是**会让人算错数字**的陷阱，不是使用说明。
> 使用说明在根目录 `CLAUDE.md`。

---

## 1. Reasoning token 占 output 的 74–97%，成本几乎全在这里

**实测日期：2026-08-23**（Google AI Studio，`.env` 当前配置的两个模型）

| 模型 | prompt 类型 | input | output | 其中 reasoning | reasoning 占 output |
|---|---|---:|---:|---:|---:|
| `gemma-4-31b-it` | trivial（"Reply with just: OK"） | 6 | 30 | 29 | **96.7%** |
| `gemma-4-31b-it` | short-qa（一句话问答） | 10 | 328 | 296 | **90.2%** |
| `gemma-4-31b-it` | synthesis（3 商品，3 句推荐） | 44 | 393 | 338 | **86.0%** |
| `gemma-4-31b-it` | long-syn（5 商品，约 150 词） | 75 | 753 | 558 | **74.1%** |
| `gemini-2.5-flash` | trivial | 6 | 20 | 19 | **95.0%** |
| `gemini-2.5-flash` | short-qa | 10 | 804 | 770 | **95.8%** |
| `gemini-2.5-flash` | synthesis | 44 | 903 | 855 | **94.7%** |
| `gemini-2.5-flash` | long-syn | 75 | 2,339 | 2,116 | **90.5%** |

数据来源：`usage_metadata.output_token_details.reasoning`。

### 为什么这会让人算错

**reasoning token 计入 `output_tokens`，按 output 单价计费。** 所以：

- 从「可见回答有多长」去估成本，会**低估一个数量级**。`gemini-2.5-flash` 回答「用一句话解释什么是极简风格」花了 **804 个 output token，其中 770 个是 reasoning** —— 用户看到的那一句话只占 4%。
- 成本的大头在 **router**，不在 finalizer。router 是 `gemini-2.5-flash`（output $2.50/M），而且 **ReAct 循环里 router 每轮迭代都跑一次**。一个 3 次迭代的 turn 会付三遍这个钱。
- 这一条**加强**了「模型分层」的论证（把合成挪到免费的 Gemma，等于免掉了 Gemma 那 74–86% 的 reasoning 费用），但同时说明**router 的单次调用远比想象中贵**，所以「减少迭代次数」是比「换更便宜的 finalizer」更有效的省钱方向。

### 延迟含义

reasoning token 要逐个生成，**是墙钟时间的主要来源**。看到某个 turn 的 `router` 节点延迟高，第一反应应该是「这次 reasoning 多」，而不是「网络慢」。分析延迟异常时，先看该节点的 `output_tokens`。

### 当前埋点的已知缺口

`agent/metrics.py` 的 `extract_usage()` 只取 `input_tokens` / `output_tokens` / `total_tokens`，**丢弃了 `output_token_details`**。

后果：`NodeMetric.output_tokens` 是「reasoning + 可见输出」的合计，**当前无法把两者拆开**。

- 计费是**对的**（reasoning 本来就按 output 计价，`total = input + output` 已验证一致）
- 但如果将来要回答「多少钱花在思考上、多少花在实际回答上」，需要先扩展 `extract_usage()` 去带上 `output_token_details.reasoning`

**在扩展之前，不要声称任何「reasoning 成本占比」的数字来自 `turns.jsonl`——那个文件里没有这个字段。** 上表的数字来自一次性的独立探针，不是生产埋点。

---

## 1b. 成本和延迟由不同的模型层主导，方向相反（实测 n=42，付费层）

| 节点 | 延迟占比 | 成本占比 | token 占比 |
|---|---:|---:|---:|
| `finalizer`（gemma-4-31b-it，免费） | **86.1%** | **0%** | 19.3% |
| `router`（gemini-2.5-flash，付费） | 12.5% | **100%** | 80.7% |
| `tools`（Gorse / Tavily） | 1.4% | 0% | — |

**要省钱就减少 router 迭代次数；要降延迟就得动 finalizer。这两个方向互相冲突**，因为便宜的那一层正好是慢的那一层。做任何优化决策前先想清楚在优化哪一个。

另外注意 token 份额和成本份额不成比例：finalizer 占 19.3% 的 token，但若改用 flash 跑它会占 44% 的成本。原因是 **output 单价是 input 的 8.33 倍**，而 finalizer 是 output-heavy（27.5k in / 33.9k out），router 是 input-heavy（237k in / 18.6k out）。**引用「分层省了 44%」时必须同时给出 token 份额，否则是误导。**

### 免费层 vs 付费层：Gemma 被限流约 2 倍

| | 免费层 (n=9) | 付费层 (n=42) |
|---|---:|---:|
| finalizer 延迟中位 | 48.8s | **21.0s** |
| finalizer 吞吐中位 | 16.0 tok/s | **33.8 tok/s** |
| 端到端 p50 | 47.6s | **24.8s** |

Gemma 自己没有付费层（见 `pricing.json`），但**项目开了 billing 之后它的吞吐翻倍**。

> **踩过的坑：不要用短 prompt 测吞吐。** 一次「回复 OK」的探针显示 ~15 tok/s，看起来正好能解释免费层的 80 秒，于是得出了「这是模型本身的速度、不是限流」的错误结论。实际上 35 个 token 的响应里固定开销（TTFT + 网络）占了大头，把吞吐算低了一倍多。**用真实长度的输出测，或者显式扣掉 TTFT。**

## 2. 硬编码的默认模型全部 404（已修，但记住成因）

**实测日期：2026-08-23**

```
gemma-3-27b-it   →  404 NOT_FOUND
gemma-3-12b-it   →  404 NOT_FOUND
gemma-4-31b-it   →  OK
gemini-2.5-flash →  OK
```

这两个死 id 曾出现在 6 处（`AgentConfig.final_model`、`Settings.agent_final_model`、`test_connections.py` 兜底、`.env.example` ×2、两处 docstring），只有本地 `.env` 的覆盖值是活的。任何人照 `.env.example` 配置，每次 final answer 都会 404。

**教训：Google AI Studio 会下架模型 id。** 代码里的默认值有保质期，`.env` 能跑不代表默认路径能跑。以后加新模型默认值时，顺手在 `test_connections.py` 里覆盖它——目前那个 smoke test 只打 `AGENT_ROUTER_MODEL`，finalizer 从来没被验证过，这正是这个 bug 藏了这么久的原因。

`agent/pricing.json` 里仍给这两个死 id 标着价，**无害但别当作它们可用的证据**。

---

## 3. `content` 可能是 list 而不是 str

`gemma-4-31b-it` 和 `gemini-2.5-flash` 都可能把 `AIMessage.content` 返回成 content-block **列表**（thinking block + text block），而不是字符串。

`AgentGraph.chat()` 已正确处理（过滤 `type == "thinking"` 后拼接 text）。**写任何直接读 `resp.content` 的探针脚本或新节点时要记住这点**——直接 `.strip()` 会抛 `AttributeError: 'list' object has no attribute 'strip'`。

---

## 4. MOCK_AI=true 的运行不能用于成本分析

mock 模型不带 `usage_metadata`，按设计会记成 `usage_available: false` 且 `cost_usd: null`（**不是 0.0**）。

所以 `MOCK_AI=true` 跑出来的 `turns.jsonl` 里，延迟数据有意义（框架开销真实），**token 和成本全是 null**。聚合脚本会打印 coverage 警告。看到「成本为 null」先查 `.env` 的 `MOCK_AI`，别去查价格表。
