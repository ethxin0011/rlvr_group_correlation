"""
Stage 3 / step A: select a BUDGET-CAPPED subsample from the gold CSV to send
through CompassVerifier-3B on the T4.

Why this exists: at $15 remaining T4 budget (~19-28h at $0.53-0.77/hr), the
originally planned 80k-trace CompassVerifier pass is not affordable. This
script picks the highest-value subset instead of a random one: it prioritizes
'disagreement' rows (where rule verifiers already disagree) over
'all_agree_*' rows, because disagreement rows are what actually need a
third, model-based opinion to adjudicate. Agreement rows are cheap to infer
(CompassVerifier will almost certainly agree too) and add little information
per dollar.

Input: gold_set_1p5b.csv (job 08 output).
Output: a smaller CSV, same schema, ready for compass_verify.py.
"""

import argparse
import csv
import random
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget_n", type=int, default=4000,
                    help="total traces to send through CompassVerifier")
    ap.add_argument("--disagreement_frac", type=float, default=0.7,
                    help="fraction of budget reserved for disagreement rows")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.gold_csv, encoding="utf-8")))
    print(f"[select] loaded {len(rows)} rows from {args.gold_csv}")

    by_pattern = defaultdict(list)
    for r in rows:
        by_pattern[r["agreement_pattern"]].append(r)
    print("[select] pattern counts:", {k: len(v) for k, v in by_pattern.items()})

    rng = random.Random(args.seed)
    n_disagree = int(args.budget_n * args.disagreement_frac)
    n_agree = args.budget_n - n_disagree

    disagree_rows = by_pattern.get("disagreement", [])
    rng.shuffle(disagree_rows)
    chosen = disagree_rows[:n_disagree]

    agree_rows = by_pattern.get("all_agree_correct", []) + by_pattern.get("all_agree_incorrect", [])
    rng.shuffle(agree_rows)
    # stratify the agree budget evenly across categories present
    by_cat = defaultdict(list)
    for r in agree_rows:
        by_cat[r["primary_cat"]].append(r)
    cats = list(by_cat.keys())
    per_cat = max(1, n_agree // max(len(cats), 1))
    for c in cats:
        chosen.extend(by_cat[c][:per_cat])

    rng.shuffle(chosen)
    chosen = chosen[:args.budget_n]

    print(f"[select] selected {len(chosen)} traces "
          f"(disagreement={sum(1 for r in chosen if r['agreement_pattern']=='disagreement')}, "
          f"agree={sum(1 for r in chosen if r['agreement_pattern']!='disagreement')})")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in chosen:
            w.writerow(r)
    print(f"[select] wrote -> {args.out}")


if __name__ == "__main__":
    main()
