"""Drive a batch of real agent turns to populate metrics/turns.jsonl.

Sequential on purpose: concurrent requests contend on the same free-tier
finalizer quota, which would inflate the latency distribution this run exists
to measure. A baseline has to be measured under the conditions it claims.

Run (agent must already be serving on :5002):
    python eval/run_turns.py
    python eval/run_turns.py --repeat 2      # two passes for tighter percentiles
    python eval/run_turns.py --only hitl     # one scenario group
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any

import httpx

BASE = "http://127.0.0.1:5002/api/ai"
# Generous: the free-tier finalizer has been observed at 80s+ for one call.
TIMEOUT = httpx.Timeout(600.0, connect=10.0)


# Each scenario is (group, user_id, [messages...]). Messages in one scenario
# share a session id, so later turns exercise multi-turn context growth.
SCENARIOS: list[tuple[str, str, list[str]]] = [
    ("grounded", "user_001", [
        "Show me actual product recommendations from your catalogue.",
        "Which of those is the cheapest?",
    ]),
    ("grounded", "user_002", [
        "Recommend some products I might like based on my history.",
    ]),
    ("prefs", "user_001", [
        "What style preferences do you have on file for me?",
    ]),
    ("prefs", "user_003", [
        "Do you know anything about what I like to wear?",
    ]),
    ("item", "user_001", [
        "Tell me about the product with id B07CN5QGVY.",
    ]),
    ("trends", "user_002", [
        "Search the web and tell me the biggest fashion trends right now.",
        "How would I adapt that for an office dress code?",
    ]),
    ("trends", "user_004", [
        "What is trending in sustainable fashion this season? Please search.",
    ]),
    ("hitl_approve", "user_005", [
        "I really love minimalist style and neutral colours like beige and white.",
    ]),
    ("hitl_reject", "user_006", [
        "Actually I prefer bold streetwear and very bright colours.",
    ]),
    ("nosignal", "user_001", [
        "asdfgh qwerty zxcvb",
    ]),
    ("general", "user_003", [
        "What exactly does smart casual mean?",
    ]),
    ("general", "user_004", [
        "How should I care for a linen blazer?",
    ]),
]


def one_turn(
    client: httpx.Client, user_id: str, session_id: str, message: str
) -> dict[str, Any] | None:
    t0 = time.perf_counter()
    try:
        r = client.post(
            f"{BASE}/agent-chat",
            headers={"X-User-ID": user_id, "X-Session-ID": session_id},
            json={"message": message},
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        # Keep going: one failed turn must not abandon the rest of the batch.
        print(f"    ERROR after {time.perf_counter()-t0:.1f}s: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None


def resume(client: httpx.Client, session_id: str, approved: bool) -> None:
    try:
        r = client.post(
            f"{BASE}/agent-resume",
            headers={"X-Session-ID": session_id},
            json={"approved": approved},
        )
        r.raise_for_status()
        print(f"    resume(approved={approved}) -> {r.status_code}", flush=True)
    except Exception as exc:
        print(f"    resume ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="passes over the scenario list")
    ap.add_argument("--only", help="run only scenarios in this group")
    args = ap.parse_args()

    scenarios = [s for s in SCENARIOS if not args.only or s[0] == args.only]
    n_turns = sum(len(s[2]) for s in scenarios) * args.repeat
    print(f"{n_turns} turns across {len(scenarios)} scenarios x {args.repeat} pass(es)\n", flush=True)

    done = 0
    t_start = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        for p in range(args.repeat):
            for group, user_id, messages in scenarios:
                session_id = f"{group}-{p}-{uuid.uuid4().hex[:8]}"
                for msg in messages:
                    done += 1
                    print(f"[{done}/{n_turns}] {group:<13} {user_id}  {msg[:52]!r}", flush=True)
                    resp = one_turn(client, user_id, session_id, msg)
                    if resp is None:
                        continue
                    print(
                        f"    iters={resp['iterations']:<2} "
                        f"latency={resp['latency_ms']/1000:>6.1f}s "
                        f"tokens={resp['tokens_used']:<6} "
                        f"cost=${resp['cost_usd']} "
                        f"pending={resp['pending_approval']}",
                        flush=True,
                    )
                    if resp.get("pending_approval"):
                        resume(client, session_id, approved=(group != "hitl_reject"))

    mins = (time.perf_counter() - t_start) / 60
    print(f"\nBatch finished in {mins:.1f} min.", flush=True)
    print("Aggregate with:  python eval/aggregate_metrics.py --turn-type chat", flush=True)


if __name__ == "__main__":
    main()
