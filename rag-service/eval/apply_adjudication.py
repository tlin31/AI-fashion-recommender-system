# Merges labeled candidates from eval/adjudication_candidates.json back into
# eval/golden_queries.json. Only updates queries where at least one candidate
# has been labeled (your_label != null). Skips null entries silently.
#
# Usage (run from rag-service/ after filling in your_label values):
#   python3 eval/apply_adjudication.py
#
# Safe to run multiple times — merges rather than overwrites existing labels.

from __future__ import annotations

import argparse
import json
from pathlib import Path

ADJUDICATION_PATH = Path("eval/adjudication_candidates.json")
GOLDEN_PATH       = Path("eval/golden_queries.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labeled candidates into the golden set")
    parser.add_argument("--input", default=str(ADJUDICATION_PATH),
                        help="Labeled candidates file (round 3 writes a separate file "
                             "so earlier rounds stay on record)")
    args = parser.parse_args()

    with open(args.input) as f:
        adjudicated: list[dict] = json.load(f)

    with open(GOLDEN_PATH) as f:
        golden: list[dict] = json.load(f)

    golden_by_id = {q["id"]: q for q in golden}

    total_labels_added = 0
    queries_updated = 0

    for entry in adjudicated:
        qid = entry["id"]
        labeled = [
            c for c in entry.get("candidates", [])
            if c.get("your_label") is not None
        ]
        if not labeled:
            print(f"  [{qid}] skipped — no labels filled in yet")
            continue

        if qid not in golden_by_id:
            print(f"  [{qid}] WARNING: not found in golden_queries.json, skipping")
            continue

        relevance: dict[str, int] = golden_by_id[qid].setdefault("relevance", {})
        added = 0
        for c in labeled:
            pid   = c["product_id"]
            label = int(c["your_label"])
            if pid not in relevance:
                relevance[pid] = label
                added += 1
            else:
                # Already labeled — keep existing value, just note the conflict
                if relevance[pid] != label:
                    print(
                        f"  [{qid}] {pid}: existing label={relevance[pid]}, "
                        f"adjudicated={label} — keeping existing"
                    )

        if added:
            queries_updated += 1
            total_labels_added += added
            print(f"  [{qid}] +{added} new labels  (relevance dict now has {len(relevance)} entries)")
        else:
            print(f"  [{qid}] all candidates already labeled, nothing new added")

    # Write updated golden queries back
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {total_labels_added} new labels merged across {queries_updated} queries.")
    print(f"Updated: {GOLDEN_PATH}")
    print(f"\nRerun eval to see the impact:")
    print(f"  python3 eval/run_eval.py --golden-set eval/golden_queries.json --skip-faithfulness")


if __name__ == "__main__":
    main()
