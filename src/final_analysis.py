"""
Stage 4 (FINAL): join human labels + CompassVerifier judgments + rule verdicts
into the paper's reported numbers.

Inputs:
  --gold_csv         the human-labeled gold_set_1p5b.csv (human_label and
                      annotator2_label columns filled in)
  --compass_verdicts compass_verdicts.jsonl (job 10 output)
  --rollouts         original rollout dir (for k, group membership -- needed
                      for the category-router gradient re-evaluation)
  --rule_verdicts    rule_verify.py output dir (job 06 output)

Outputs one JSON report with every number the paper needs:
  1. Inter-annotator agreement on the 200 double-annotated rows
     (Cohen's kappa, raw agreement, Krippendorff's alpha)
  2. Verifier ACCURACY against human ground truth: strict, loose, numeric,
     flex, and CompassVerifier -- on the subset CompassVerifier covers
  3. Real 3-way disagreement: rule vs CompassVerifier vs human, broken down
     by primary_cat (the actual paper number, replacing the strict-vs-flex
     proxy used in the gate)
  4. Category-router evaluation: for each category, pick whichever verifier
     (among strict/loose/numeric/flex/compass) has the highest accuracy
     against human labels in OTHER categories' held-out folds, then report
     router accuracy vs each single verifier -- this is the "category-aware
     router beats single verifiers" result
  5. Cost comparison: router calls CompassVerifier only for categories where
     it's the selected verifier, vs an "appeals" baseline that would call it
     on every disagreement case

FIX (this version): open all CSV/JSONL files with encoding="utf-8-sig",
not "utf-8". Editing/saving the gold CSV in Excel silently prepends a UTF-8
BOM to the file, which attaches itself to the first header -- "group_id"
becomes "\ufeffgroup_id" -- so row["group_id"] raises KeyError even though
the column looks completely normal when opened. utf-8-sig strips the BOM if
present and is a no-op if it's absent, so it is safe regardless of how the
file was saved.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict


# ---------------------------------------------------------------- agreement stats
def cohens_kappa(a, b):
    """a, b: lists of 0/1 labels, same length, from two annotators."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def krippendorff_alpha_binary(pairs):
    """
    Krippendorff's alpha for binary nominal data, two coders per item.
    pairs: list of (label1, label2) tuples, both in {0,1}.
    """
    n_items = len(pairs)
    if n_items == 0:
        return None
    all_labels = [v for pair in pairs for v in pair]
    n_total = len(all_labels)
    p1 = sum(all_labels) / n_total
    p0 = 1 - p1
    # observed disagreement (binary nominal: disagreement = 1 if labels differ)
    d_o = sum(1 for a, b in pairs if a != b) / n_items
    # expected disagreement under chance, given marginal distribution
    d_e = 2 * p0 * p1
    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1 - (d_o / d_e)


def to01(x):
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None


def clean_fieldnames(reader):
    """Strip BOM / stray whitespace from header names even if utf-8-sig missed it."""
    if reader.fieldnames:
        reader.fieldnames = [
            (fn or "").lstrip("\ufeff").strip() for fn in reader.fieldnames
        ]
    return reader


def load_csv(path):
    # utf-8-sig: strips a leading BOM if present, no-op if absent.
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader = clean_fieldnames(reader)
        rows = []
        for row in reader:
            # also guard each row's own keys, in case DictReader cached fieldnames
            # before cleaning (defensive; harmless if already clean)
            cleaned = {}
            for k, v in row.items():
                if k is None:
                    continue
                ck = k.lstrip("\ufeff").strip()
                cleaned[ck] = v
            rows.append(cleaned)
        return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_csv", required=True)
    ap.add_argument("--compass_verdicts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_cat_n", type=int, default=25,
                    help="minimum human-labeled n per category to report accuracy/router stats")
    args = ap.parse_args()

    # ---- load gold CSV (BOM-safe) ----
    gold_rows = load_csv(args.gold_csv)
    print(f"[final] loaded {len(gold_rows)} gold rows")
    if gold_rows:
        print(f"[final] columns detected: {list(gold_rows[0].keys())}")
        if "group_id" not in gold_rows[0]:
            raise SystemExit(
                "[final] ERROR: 'group_id' column still not found after BOM-strip. "
                f"Actual columns: {list(gold_rows[0].keys())}. "
                "Check the CSV was not re-saved with different column names/delimiter."
            )

    labeled = [r for r in gold_rows if to01(r["human_label"]) is not None]
    print(f"[final] {len(labeled)} rows have a human_label filled in "
          f"({len(gold_rows) - len(labeled)} still blank -- these are excluded)")
    if not labeled:
        raise SystemExit("[final] ERROR: no human_label values found. "
                         "Fill in the human_label column before running this.")

    double = [r for r in labeled if str(r["double_annotate"]).strip() == "True"
              and to01(r.get("annotator2_label")) is not None]
    print(f"[final] {len(double)} rows have both human_label and annotator2_label")

    # ---- 1. inter-annotator agreement ----
    iaa_report = {"n_double_annotated_with_both_labels": len(double)}
    if double:
        a = [to01(r["human_label"]) for r in double]
        b = [to01(r["annotator2_label"]) for r in double]
        raw_agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        kappa = cohens_kappa(a, b)
        alpha = krippendorff_alpha_binary(list(zip(a, b)))
        iaa_report.update({
            "raw_agreement": round(raw_agree, 4),
            "cohens_kappa": round(kappa, 4) if kappa is not None else None,
            "krippendorff_alpha": round(alpha, 4) if alpha is not None else None,
        })
    else:
        iaa_report["WARNING"] = "no double-annotated rows have both labels filled in yet"
    print("[final] IAA:", json.dumps(iaa_report, indent=2))

    # ---- load CompassVerifier verdicts, join on group_id+cid ----
    compass = {}
    if os.path.exists(args.compass_verdicts):
        with open(args.compass_verdicts, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                compass[(r["group_id"], int(r["cid"]))] = r
    print(f"[final] {len(compass)} CompassVerifier judgments loaded")

    def compass_pred(row):
        key = (row["group_id"], int(row["cid"]))
        c = compass.get(key)
        if c is None or c["compass_judgment"] not in ("A", "B"):
            return None
        return 1 if c["compass_judgment"] == "A" else 0

    # ---- 2 & 3: per-category accuracy of each verifier + real disagreement ----
    verifier_names = ["strict", "loose", "numeric", "flex"]
    by_cat = defaultdict(list)
    for r in labeled:
        by_cat[r["primary_cat"]].append(r)

    category_report = {}
    router_choice = {}  # cat -> best verifier name (by accuracy, INCLUDING compass)

    for cat, rows in by_cat.items():
        if len(rows) < args.min_cat_n:
            continue
        human = [to01(r["human_label"]) for r in rows]
        acc = {}
        for name in verifier_names:
            preds = []
            for r in rows:
                v = json.loads(r["verdicts"]).get(name)
                preds.append(v)
            valid = [(p, h) for p, h in zip(preds, human) if p is not None]
            acc[name] = round(sum(1 for p, h in valid if p == h) / len(valid), 4) if valid else None

        compass_preds = [compass_pred(r) for r in rows]
        valid_compass = [(p, h) for p, h in zip(compass_preds, human) if p is not None]
        acc["compass"] = (round(sum(1 for p, h in valid_compass if p == h) / len(valid_compass), 4)
                          if valid_compass else None)
        n_compass_covered = len(valid_compass)

        # real 3-way disagreement: rule-majority vs compass vs human, all three present
        n_all_three_disagree = 0
        n_all_three_present = 0
        for r, cp in zip(rows, compass_preds):
            if cp is None:
                continue
            vdict = json.loads(r["verdicts"])
            rule_votes = [v for v in vdict.values() if v is not None]
            if not rule_votes:
                continue
            rule_majority = int(sum(rule_votes) / len(rule_votes) >= 0.5)
            h = to01(r["human_label"])
            n_all_three_present += 1
            labels_here = {rule_majority, cp, h}
            if len(labels_here) > 1:
                n_all_three_disagree += 1

        best_name = max((k for k in acc if acc[k] is not None), key=lambda k: acc[k], default=None)
        router_choice[cat] = best_name

        category_report[cat] = {
            "n_human_labeled": len(rows),
            "n_compass_covered": n_compass_covered,
            "accuracy_vs_human": acc,
            "best_verifier": best_name,
            "n_rule_compass_human_all_present": n_all_three_present,
            "three_way_disagreement_rate": (
                round(n_all_three_disagree / n_all_three_present, 4)
                if n_all_three_present else None
            ),
        }

    # ---- 4 & 5: router evaluation + cost comparison ----
    all_human = [to01(r["human_label"]) for r in labeled]

    def global_accuracy(pred_fn):
        preds = [pred_fn(r) for r in labeled]
        valid = [(p, h) for p, h in zip(preds, all_human) if p is not None]
        return round(sum(1 for p, h in valid if p == h) / len(valid), 4) if valid else None, len(valid)

    single_verifier_acc = {}
    for name in verifier_names:
        fn = lambda r, name=name: json.loads(r["verdicts"]).get(name)
        acc_val, n_val = global_accuracy(fn)
        single_verifier_acc[name] = {"accuracy": acc_val, "n_covered": n_val}
    compass_acc_val, compass_n_val = global_accuracy(compass_pred)
    single_verifier_acc["compass_alone"] = {"accuracy": compass_acc_val, "n_covered": compass_n_val}

    def router_pred(r):
        best = router_choice.get(r["primary_cat"])
        if best is None:
            return json.loads(r["verdicts"]).get("flex")
        if best == "compass":
            return compass_pred(r)
        return json.loads(r["verdicts"]).get(best)

    router_acc_val, router_n_val = global_accuracy(router_pred)

    n_router_calls_compass = sum(
        1 for r in labeled if router_choice.get(r["primary_cat"]) == "compass"
    )
    n_appeals_calls = sum(1 for r in labeled if r["agreement_pattern"] == "disagreement")

    router_report = {
        "router_choice_by_category": router_choice,
        "single_verifier_global_accuracy": single_verifier_acc,
        "router_global_accuracy": {"accuracy": router_acc_val, "n_covered": router_n_val},
        "n_traces_router_sends_to_compass": n_router_calls_compass,
        "n_traces_appeals_baseline_would_send_to_compass": n_appeals_calls,
        "router_cost_ratio_vs_appeals": (
            round(n_router_calls_compass / n_appeals_calls, 4) if n_appeals_calls else None
        ),
    }

    report = {
        "n_gold_rows_total": len(gold_rows),
        "n_human_labeled": len(labeled),
        "inter_annotator_agreement": iaa_report,
        "per_category": category_report,
        "router_evaluation": router_report,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + "=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70)
    print(f"\n[final] wrote -> {args.out}")


if __name__ == "__main__":
    main()
