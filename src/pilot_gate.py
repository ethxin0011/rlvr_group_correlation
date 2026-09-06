"""
Stage 1 / step D: THE HARD GATE.

Run on the pilot / early-real rollouts BEFORE committing further GPU time.
All three gates must pass or the paper has nothing to measure.

  GATE 1  mixed-group rate >= 0.30
          Groups where the cheap rule-verifier scores neither all-correct nor
          all-wrong. Degenerate groups give zero advantage to every rollout, so
          there is no learning signal for verifier choice to corrupt.
          Calibration: 0.69 degeneracy reported at G=4; k=8 should do better.

  GATE 2  0 < within-group category diversity, and mean primary-cat share < 0.98
          The paradox: TOTAL format collapse makes every rollout identical, so
          rho is trivially 1 and uninformative. We need SHARED BUT NOT IDENTICAL
          surface form. Both extremes fail.

  GATE 3  verifier disagreement >= 0.02
          Below ~2% the 1.5k human gold set has too few disagreement cases per
          stratum to carry bootstrap CIs.

          HISTORY OF THIS GATE (read before changing thresholds again):
          v1 compared strict vs loose -- both string-normalizers, near-zero
          disagreement once answers are cleanly extracted from \\boxed{}.
          v2 compared strict vs numeric (float, relative-tolerance) -- but
          v_numeric() returns None (inapplicable) for anything that isn't a
          bare number, which EXCLUDES exactly the categories that dominate
          real data: degree_percent ("50%" vs "0.5"), interval_or_tuple
          ("(1,3)" vs "(1, 3)"), set_or_list ("{1,2}" vs "{2,1}"). Those groups
          never contributed to the disagreement count at all.
          v3 (this version) adds v_flex(): percent<->decimal equivalence,
          a/b fraction parsing, and order-insensitive numeric tuple/set/list
          comparison with tolerance. This reaches the categories v2 skipped
          and is the closest cheap proxy for real rule-vs-model divergence
          without running an actual second verifier.

Also emits a *preliminary* rho_cat using strict-vs-flex agreement as a proxy
error indicator. This is NOT the paper number (that needs adjudication against
CompassVerifier / an LLM judge in Stage 3-4), but it is materially more
representative than either prior version.

Exit codes:
    0  all gates pass
    2  at least one gate failed  (this is a RESULT, not a crash)
    1  anything else went wrong
"""

import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------- verifiers
def norm_loose(s):
    if s is None:
        return ""
    s = s.strip().rstrip(".").replace(" ", "").replace(",", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s


def norm_strict(s):
    return "" if s is None else s.strip()


def v_strict(pred, gt):
    return int(norm_strict(pred) == norm_strict(gt))


def v_loose(pred, gt):
    return int(norm_loose(pred) == norm_loose(gt))


def _parse_number(s):
    """Bare float, or a/b fraction. Returns float or None."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^-?\d+(\.\d+)?\s*/\s*-?\d+(\.\d+)?$", s)
    if m:
        try:
            a, b = s.split("/")
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    m = re.match(r"^\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    return None


def _as_value(s):
    """Number, with percent handling: '50%' -> 0.5. Returns float or None."""
    s = norm_loose(s)
    had_pct = s.endswith("%") or s.endswith("\\%")
    core = s.rstrip("%").rstrip("\\")
    v = _parse_number(core)
    if v is None:
        return None
    return v / 100.0 if had_pct else v


def _close(a, b, rel_tol=1e-4):
    if b == 0:
        return abs(a) < rel_tol
    return abs(a - b) / abs(b) < rel_tol


def v_numeric(pred, gt, rel_tol=1e-4):
    """Bare-number relative-tolerance comparison. None if either side isn't a plain number."""
    a, b = _as_value(pred), _as_value(gt)
    if a is None or b is None:
        return None
    return int(_close(a, b, rel_tol))


def _extract_number_list(s):
    """
    '(1, 3)' / '{2,1}' / '[1,2,3]' -> list of floats, order preserved.
    None if any part fails to parse.

    IMPORTANT: strip only bracket characters here, do NOT call norm_loose()
    first -- norm_loose() strips commas, which destroys the delimiter we
    need to split on (e.g. "(3,1)" would collapse to "31" as one number).
    """
    s = s.strip().strip("{}[]() ")
    if not s:
        return None
    parts = re.split(r"[,;]\s*", s)
    vals = []
    for p in parts:
        v = _as_value(p)
        if v is None:
            return None
        vals.append(v)
    return vals


def v_flex(pred, gt, rel_tol=1e-4):
    """
    Closest cheap proxy for real rule-vs-model verifier disagreement.
    Handles, in order:
      1. bare numbers / fractions (delegates to v_numeric's logic)
      2. percent <-> decimal equivalence ("50%" == "0.5")
      3. order-insensitive numeric tuple/set/list ("(1,3)" == "(3, 1)")
    Returns 1/0, or None if pred/gt don't fit any of these forms (falls back
    to strict-vs-loose for that group so it still contributes to rho_cat).
    """
    a, b = _as_value(pred), _as_value(gt)
    if a is not None and b is not None:
        return int(_close(a, b, rel_tol))

    pl, gl = _extract_number_list(pred), _extract_number_list(gt)
    if pl is not None and gl is not None and len(pl) == len(gl) and len(pl) > 0:
        remaining = list(gl)
        for v in pl:
            match = next((i for i, g2 in enumerate(remaining) if _close(v, g2, rel_tol)), None)
            if match is None:
                return 0
            remaining.pop(match)
        return 1

    return None


VERIFIERS = {"strict": v_strict, "loose": v_loose, "numeric": v_numeric, "flex": v_flex}


# ---------------------------------------------------------------- statistics
def icc_binary(groups_of_errors):
    """
    One-way random-effects ICC(1) on binary error indicators.
    groups_of_errors: list of lists, each inner list = error flags in one group.
    Returns (icc, n_eff, k_bar) or (None, None, None) if undefined.
    """
    groups = [g for g in groups_of_errors if len(g) > 1]
    if len(groups) < 10:
        return None, None, None

    N = sum(len(g) for g in groups)
    m = len(groups)
    grand = sum(sum(g) for g in groups) / N

    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)

    df_b, df_w = m - 1, N - m
    if df_b <= 0 or df_w <= 0:
        return None, None, None
    msb, msw = ssb / df_b, ssw / df_w

    k_bar = (N - sum(len(g) ** 2 for g in groups) / N) / (m - 1)
    denom = msb + (k_bar - 1) * msw
    if denom == 0:
        return 0.0, k_bar, k_bar
    icc = (msb - msw) / denom
    icc = max(-1.0, min(1.0, icc))
    n_eff = k_bar / (1 + (k_bar - 1) * max(icc, 0.0))  # Kish design effect
    return icc, n_eff, k_bar


def bootstrap_icc(groups_of_errors, n_boot=400, seed=7):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(groups_of_errors) for _ in groups_of_errors]
        v, _, _ = icc_binary(samp)
        if v is not None:
            vals.append(v)
    if len(vals) < 20:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


# ---------------------------------------------------------------- io
def resolve_files(path_arg):
    """
    Accepts a DIRECTORY (recurses for *.jsonl), a GLOB pattern, or a single FILE.
    Never returns a directory -- that was the earlier IsADirectoryError bug.
    """
    if os.path.isdir(path_arg):
        found = glob.glob(os.path.join(path_arg, "**", "*.jsonl"), recursive=True)
    else:
        found = glob.glob(path_arg)
        if not found:
            found = glob.glob(os.path.join(path_arg.rstrip("/"), "*.jsonl"))
    files = [f for f in found if os.path.isfile(f) and f.endswith(".jsonl")]
    return sorted(files)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="dir, glob, or file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_mixed", type=float, default=0.30)
    ap.add_argument("--max_cat_share", type=float, default=0.98)
    ap.add_argument("--min_disagree", type=float, default=0.02)
    ap.add_argument("--only_model", default=None,
                    help="keep only groups whose 'model' field matches this")
    args = ap.parse_args()

    files = resolve_files(args.rollouts)
    print(f"[gate] input arg: {args.rollouts}")
    print(f"[gate] resolved {len(files)} jsonl file(s)")
    for f in files:
        print(f"[gate]   {os.path.basename(f)}")
    if not files:
        raise SystemExit("[gate] ERROR: no .jsonl files found")

    groups = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                groups.append(json.loads(line))

    if args.only_model:
        before = len(groups)
        groups = [g for g in groups if g.get("model") == args.only_model]
        print(f"[gate] filtered to model={args.only_model}: {before} -> {len(groups)}")

    print(f"[gate] loaded {len(groups)} groups")
    if not groups:
        raise SystemExit("[gate] ERROR: no groups loaded")

    models_present = Counter(g.get("model", "?") for g in groups)
    print(f"[gate] models present: {dict(models_present)}")
    if len(models_present) > 1:
        print("[gate] WARNING: multiple models pooled. Different models have "
              "different format habits -- consider --only_model.")

    mixed = 0
    cat_shares = []
    n_uniform_cat = 0
    disagree_n = disagree_d = 0
    n_flex_inapplicable = 0
    err_by_group = []
    err_by_group_by_cat = defaultdict(list)
    primary_counter = Counter()
    by_model = defaultdict(lambda: {"mixed": 0, "n": 0, "errs": []})

    for g in groups:
        gt = g["gt_answer"]
        comps = g["completions"]
        model = g.get("model", "?")

        r_loose = [v_loose(c["raw_answer"], gt) for c in comps]
        r_strict = [v_strict(c["raw_answer"], gt) for c in comps]
        r_flex = [v_flex(c["raw_answer"], gt) for c in comps]

        # GATE 1 (mixed-group rate; based on the lenient rule verifier)
        is_mixed = 0 < sum(r_loose) < len(r_loose)
        mixed += int(is_mixed)
        by_model[model]["n"] += 1
        by_model[model]["mixed"] += int(is_mixed)

        # GATE 2 (category diversity)
        cats = [c["primary_cat"] for c in comps]
        primary_counter.update(cats)
        share = Counter(cats).most_common(1)[0][1] / len(cats)
        cat_shares.append(share)
        if share == 1.0:
            n_uniform_cat += 1

        # GATE 3 -- STRICT vs FLEX. Flex covers bare numbers, percent<->decimal,
        # fractions, and order-insensitive numeric tuples/sets/lists -- i.e. it
        # actually reaches degree_percent / interval_or_tuple / set_or_list,
        # which pure v_numeric skipped entirely (see module docstring).
        applicable = [(s, fx) for s, fx in zip(r_strict, r_flex) if fx is not None]
        if len(applicable) < len(comps):
            n_flex_inapplicable += 1
        for s, fx in applicable:
            disagree_d += 1
            disagree_n += int(s != fx)

        # preliminary rho proxy: strict-verifier error vs flex "reference truth"
        if applicable and len(applicable) == len(comps):
            errs = [int(s != fx) for s, fx in applicable]
        else:
            # fallback for groups flex can't parse at all (rare symbolic answers):
            # use strict vs loose so every group still contributes to rho_cat.
            errs = [int(s != l) for s, l in zip(r_strict, r_loose)]

        err_by_group.append(errs)
        by_model[model]["errs"].append(errs)
        dom = Counter(cats).most_common(1)[0][0]
        err_by_group_by_cat[dom].append(errs)

    n = len(groups)
    mixed_rate = mixed / n
    mean_cat_share = sum(cat_shares) / n
    uniform_cat_rate = n_uniform_cat / n
    disagree_rate = disagree_n / max(disagree_d, 1)
    flex_inapplicable_rate = n_flex_inapplicable / n

    icc, n_eff, k_bar = icc_binary(err_by_group)
    lo, hi = bootstrap_icc(err_by_group) if icc is not None else (None, None)

    per_cat = {}
    for cat, gl in err_by_group_by_cat.items():
        if len(gl) >= 25:
            v, ne, _ = icc_binary(gl)
            if v is not None:
                per_cat[cat] = {"n_groups": len(gl), "icc": round(v, 4),
                                "n_eff": round(ne, 3)}

    g1 = mixed_rate >= args.min_mixed
    g2 = (mean_cat_share < args.max_cat_share) and (uniform_cat_rate < 1.0)
    g3 = disagree_rate >= args.min_disagree

    report = {
        "n_groups": n,
        "n_files": len(files),
        "models_present": dict(models_present),
        "gate1_mixed_group_rate": round(mixed_rate, 4),
        "gate1_threshold": args.min_mixed,
        "gate1_pass": g1,
        "gate2_mean_primary_cat_share": round(mean_cat_share, 4),
        "gate2_fully_uniform_cat_rate": round(uniform_cat_rate, 4),
        "gate2_threshold_max_share": args.max_cat_share,
        "gate2_pass": g2,
        "gate3_verifier_disagreement_rate": round(disagree_rate, 4),
        "gate3_disagree_pairs_counted": disagree_d,
        "gate3_comparison": "strict_vs_flex",
        "gate3_flex_inapplicable_group_rate": round(flex_inapplicable_rate, 4),
        "gate3_threshold": args.min_disagree,
        "gate3_pass": g3,
        "ALL_GATES_PASS": bool(g1 and g2 and g3),
        "preliminary_rho_cat_proxy": None if icc is None else round(icc, 4),
        "preliminary_rho_ci95": None if lo is None else [round(lo, 4), round(hi, 4)],
        "kish_n_eff_of_k": None if n_eff is None else round(n_eff, 3),
        "k_bar": None if k_bar is None else round(k_bar, 2),
        "rho_by_primary_cat": per_cat,
        "primary_cat_distribution": dict(primary_counter.most_common(15)),
        "per_model_mixed_rate": {
            m: round(v["mixed"] / max(v["n"], 1), 4) for m, v in by_model.items()
        },
        "per_model_rho": {
            m: (lambda r: None if r[0] is None else round(r[0], 4))(icc_binary(v["errs"]))
            for m, v in by_model.items()
        },
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\n" + "=" * 62)
    for name, ok in (("GATE 1 mixed groups", g1),
                     ("GATE 2 category diversity", g2),
                     ("GATE 3 disagreement (strict-vs-flex)", g3)):
        print(f"  {name:<40} {'PASS' if ok else 'FAIL'}")
    print("=" * 62)
    if not (g1 and g2 and g3):
        print("DO NOT PROCEED. Adjust temperature / prompt difficulty / format "
              "instruction and re-run the pilot.")
        raise SystemExit(2)
    print("All gates clear. Proceed to full 400k generation.")


if __name__ == "__main__":
    main()
