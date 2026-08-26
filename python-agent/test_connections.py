"""Minimal connectivity smoke-test for every external service the agent uses.

Run from python-agent/:
    python3 test_connections.py

Each check is independent — a failure in one does NOT block the others.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Load .env before importing any library that reads env vars.
try:
    from dotenv import load_dotenv          # pip install python-dotenv
    load_dotenv()
except ImportError:
    # pydantic-settings can also load it; fall back to manual parse.
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


# ── colour helpers ──────────────────────────────────────────────────────────
def ok(msg: str)   -> None: print(f"  \033[32m✓\033[0m  {msg}")
def fail(msg: str) -> None: print(f"  \033[31m✗\033[0m  {msg}")
def info(msg: str) -> None: print(f"     {msg}")


# ── 1. Google AI Studio / Gemma ──────────────────────────────────────────────
def _as_text(content) -> str:
    """AIMessage.content may be a list of content blocks, not a string.

    Both gemini-2.5-flash and gemma-4-31b-it return thinking + text blocks, so
    calling .strip() on the raw value raises AttributeError. See MEMORY.md §3.
    """
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
            if not (isinstance(b, dict) and b.get("type") == "thinking")
        ).strip()
    return (content or "").strip()


async def check_openai() -> bool:
    """Probe BOTH tiered models, not just the router.

    Checking only AGENT_ROUTER_MODEL is how a dead AGENT_FINAL_MODEL default
    stayed hidden: the smoke test passed while every final answer 404'd.
    """
    print("\n[1/4] Google AI Studio — router + finalizer")
    api_key = os.environ.get("GOOGLE_API_KEY", "")

    if not api_key or api_key == "your-google-ai-studio-key-here":
        fail("GOOGLE_API_KEY is not set in .env")
        return False

    models = {
        "router":    os.environ.get("AGENT_ROUTER_MODEL", "gemini-2.5-flash"),
        "finalizer": os.environ.get("AGENT_FINAL_MODEL", "gemma-4-31b-it"),
    }

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
    except Exception as exc:
        fail(str(exc))
        return False

    all_ok = True
    for role, model in models.items():
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
            resp = await llm.ainvoke([HumanMessage(content="Reply with just the word: OK")])
            ok(f"{role:<9} model={model}  reply={_as_text(resp.content)[:20]!r}")
        except Exception as exc:
            fail(f"{role:<9} model={model}  {exc}")
            all_ok = False
    return all_ok


# ── 2. Tavily ────────────────────────────────────────────────────────────────
async def check_tavily() -> bool:
    print("\n[2/4] Tavily web search")
    tavily_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_key or tavily_key == "tvly-your-tavily-key-here":
        fail("TAVILY_API_KEY is not set in .env")
        return False

    try:
        from langchain_tavily import TavilySearch
        tool = TavilySearch(max_results=1)
        results = await tool.ainvoke({"query": "minimalist fashion 2026"})
        # results is a list of dicts or a string depending on version
        snippet = str(results)[:120].replace("\n", " ")
        ok(f"got results — {snippet}…")
        return True
    except Exception as exc:
        fail(str(exc))
        return False


# ── 3. PostgreSQL ────────────────────────────────────────────────────────────
async def check_postgres() -> bool:
    print("\n[3/4] PostgreSQL")
    dsn = os.environ.get("DATABASE_URL", "")

    if not dsn:
        fail("DATABASE_URL is not set in .env")
        return False

    import re
    host_match = re.search(r"@([^:/]+)", dsn)
    host = host_match.group(1) if host_match else "?"
    if host not in ("localhost", "127.0.0.1"):
        info(f"host={host!r} — looks like a Docker service name.")
        info("If running outside Docker this will time out; that is expected.")

    try:
        import asyncpg
        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=5.0)
        row = await conn.fetchrow("SELECT current_database(), current_user, version()")
        await conn.close()
        ok(f"db={row[0]}  user={row[1]}")
        info(row[2][:60])
        return True
    except asyncio.TimeoutError:
        fail(f"Connection to {host} timed out (5 s) — is the DB reachable from here?")
        return False
    except Exception as exc:
        fail(str(exc))
        return False


# ── 4. Gorse ────────────────────────────────────────────────────────────────
async def check_gorse() -> bool:
    print("\n[4/4] Gorse recommendation engine")
    gorse_url = os.environ.get("GORSE_URL", "http://localhost:8088")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{gorse_url}/api/dashboard/stats")
            if resp.status_code < 500:
                ok(f"reachable at {gorse_url}  (HTTP {resp.status_code})")
                return True
            else:
                fail(f"HTTP {resp.status_code} from {gorse_url}")
                return False
    except Exception as exc:
        fail(f"{gorse_url} — {exc}")
        return False


# ── main ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    print("=" * 56)
    print("  python-agent connection smoke-test")
    print("=" * 56)

    results = await asyncio.gather(
        check_openai(),
        check_tavily(),
        check_postgres(),
        check_gorse(),
        return_exceptions=True,
    )

    passed = sum(1 for r in results if r is True)
    total  = len(results)

    print("\n" + "=" * 56)
    print(f"  {passed}/{total} checks passed")
    print("=" * 56)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
