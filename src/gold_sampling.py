"""
Stage 2 / step C: stratified sampling plan for the 1,500-item human gold set.

Strata = (primary_cat, agreement_pattern), restricted to categories with
n_groups >= 100 in the pilot gate (small strata below that gave an unstable
ICC -- e.g. large_numeric flipped 0.0 -> 0.66 between n=29 and n=243).

agreement_pattern per group, using MAJORITY vote (over k completions) of each
verifier config:
  all_agree_correct     every verifier's majority vote is correct
  all_agree_incorrect   every verifier's majority vote is incorrect
  disagreement          verifiers' majority votes are NOT unanimous

Target: up to 300 traces per (category x pattern) cell, capped so
disagreement cells are prioritized when a category has fewer than 300 total.
Output: CSV ready for annotators, with verdicts shown but a blank
'human_label' column.
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict

VERIFIER_NAMES = ["strict", "loose", "numeric", "flex"]
# from the locked 1.5B gate report -- categories with n_groups >= 100
ELIGIBLE_CATS = {
    "small_numeric", "symbolic_other", "latex_frac", "trailing_whitespace",
    "latex_text_unit", "latex_sqrt", "trailing_period", "large_numeric",
    "interval_or_tuple",
}


def resolve_files(path_arg, suffix=".jsonl"):
    if os.path.isdir(path_arg):
        found = glob.glob(os.path.join(path_arg, "**", f"*{suffix}"), recursive=True)
    else:
        found = glob.glob(path_arg)
    return sorted(f for f in found if os.path.isfile(f))


def majority(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return int(sum(vals) / len(vals) >= 0.5)


def pattern_for_group(vmap_group, comps):
    """
    FIX: the previous version tried to use loop variable `c` inside a list
    comprehension that iterates over `vals` (bound to `v`) -- `c` is out of
    scope there, causing NameError. Rewritten as an explicit loop over comps.
    """
    per_verifier_majority = {}
    for name in VERIFIER_NAMES:
        vals = []
        for c in comps:
            v = vmap_group[c["cid"]].get(name)
            if v is None:
                v = vmap_group[c["cid"]].get("loose")
            vals.append(v)
        per_verifier_majority[name] = majority(vals)

    votes = set(v for v in per_verifier_majority.values() if v is not None)
    if len(votes) <= 1:
        only = next(iter(votes)) if votes else None
        if only == 1:
            return "all_agree_correct"
        if only == 0:
            return "all_agree_incorrect"
        return "undetermined"
    return "disagreement"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only_model", default=None)
    ap.add_argument("--per_cell", type=int, default=300)
    ap.add_argument("--double_annotate_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    print("[gold] loading verdicts...")
    vmap = defaultdict(dict)
    for fp in resolve_files(args.verdicts, "_verdicts.jsonl"):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if args.only_model and r["model"] != args.only_model:
                    continue
                vmap[r["group_id"]][r["cid"]] = r["verdicts"]
    print(f"[gold] {len(vmap)} groups with verdicts")

    cells = defaultdict(list)  # (cat, pattern) -> list of candidate rows
    rollout_files = [f for f in resolve_files(args.rollouts, ".jsonl") if "_verdicts" not in f]

    n_processed = n_skipped_incomplete = 0
    for fp in rollout_files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                if args.only_model and g.get("model") != args.only_model:
                    continue
                gid = g["group_id"]
                gvmap = vmap.get(gid)
                if not gvmap:
                    n_skipped_incomplete += 1
                    continue
                comps = sorted(g["completions"], key=lambda c: c["cid"])
                if not all(c["cid"] in gvmap for c in comps):
                    n_skipped_incomplete += 1
                    continue
                cat_counts = Counter(c["primary_cat"] for c in comps)
                dom_cat = cat_counts.most_common(1)[0][0]
                if dom_cat not in ELIGIBLE_CATS:
                    continue
                patt = pattern_for_group(gvmap, comps)
                if patt == "undetermined":
                    continue
                n_processed += 1
                for c in comps:
                    cells[(dom_cat, patt)].append({
                        "group_id": gid,
                        "prompt_id": g["prompt_id"],
                        "cid": c["cid"],
                        "model": g["model"],
                        "question": g["question"],
                        "gt_answer": g["gt_answer"],
                        "raw_answer": c["raw_answer"],
                        "primary_cat": dom_cat,
                        "agreement_pattern": patt,
                        "verdicts": json.dumps(gvmap[c["cid"]]),
                    })

    print(f"[gold] groups processed={n_processed} skipped(no/incomplete verdicts)={n_skipped_incomplete}")

    rng = random.Random(args.seed)
    sampled = []
    print("\n[gold] cell sizes and sample sizes:")
    for (cat, patt), rows in sorted(cells.items()):
        rng.shuffle(rows)
        # disagreement cells are the scientifically important ones -> take full quota first
        take = min(args.per_cell, len(rows))
        chosen = rows[:take]
        sampled.extend(chosen)
        print(f"  {cat:22} {patt:20} pool={len(rows):6} sampled={take}")

    rng.shuffle(sampled)
    for i, row in enumerate(sampled):
        row["double_annotate"] = i < args.double_annotate_n
        row["human_label"] = ""
        row["annotator2_label"] = "" if row["double_annotate"] else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["group_id", "prompt_id", "cid", "model", "question", "gt_answer",
                  "raw_answer", "primary_cat", "agreement_pattern", "verdicts",
                  "double_annotate", "human_label", "annotator2_label"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sampled:
            w.writerow(row)

    print(f"\n[gold] TOTAL sampled: {len(sampled)} (target ~1500)")
    print(f"[gold] double-annotate subset: {sum(r['double_annotate'] for r in sampled)}")
    print(f"[gold] wrote -> {args.out}")


if __name__ == "__main__":
    main()
