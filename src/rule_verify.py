"""
Stage 2 / step A: 4-config rule verifier pass over rollout groups.

Configs: strict (exact string), loose (whitespace/LaTeX-normalized string),
numeric (bare-number relative tolerance), flex (numeric + percent/fraction/
order-insensitive tuple-set-list -- see pilot_gate.py docstring for why flex
exists: numeric alone misses interval/percent/set categories entirely).

Output: one JSONL row per COMPLETION (not per group), keyed by group_id+cid,
so it joins back onto the original rollout files for advantage_replay.py and
gold_sampling.py.

Runs on CPU only. Safe to run in parallel with GPU generation jobs.
"""

import argparse
import glob
import json
import os
import re


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
    a, b = _as_value(pred), _as_value(gt)
    if a is None or b is None:
        return None
    return int(_close(a, b, rel_tol))


def _extract_number_list(s):
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


CONFIGS = {"strict": v_strict, "loose": v_loose, "numeric": v_numeric, "flex": v_flex}


# ---------------------------------------------------------------- io
def resolve_files(path_arg):
    if os.path.isdir(path_arg):
        found = glob.glob(os.path.join(path_arg, "**", "*.jsonl"), recursive=True)
    else:
        found = glob.glob(path_arg) or glob.glob(
            os.path.join(path_arg.rstrip("/"), "*.jsonl")
        )
    return sorted(f for f in found if os.path.isfile(f) and f.endswith(".jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only_model", default=None)
    args = ap.parse_args()

    files = resolve_files(args.rollouts)
    print(f"[verify] {len(files)} input file(s)")
    os.makedirs(args.out_dir, exist_ok=True)

    n_groups = n_completions = 0
    for fp in files:
        base = os.path.basename(fp)
        done_flag = os.path.join(args.out_dir, base + ".verify.done")
        out_path = os.path.join(args.out_dir, base.replace(".jsonl", "_verdicts.jsonl"))
        if os.path.exists(done_flag):
            print(f"[verify] {base} already done, skipping")
            continue

        kept_g = kept_c = 0
        with open(fp, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                if args.only_model and g.get("model") != args.only_model:
                    continue
                gt = g["gt_answer"]
                kept_g += 1
                for c in g["completions"]:
                    raw = c["raw_answer"]
                    verdicts = {name: fn(raw, gt) for name, fn in CONFIGS.items()}
                    fout.write(json.dumps({
                        "group_id": g["group_id"],
                        "prompt_id": g["prompt_id"],
                        "cid": c["cid"],
                        "model": g["model"],
                        "primary_cat": c["primary_cat"],
                        "categories": c["categories"],
                        "verdicts": verdicts,
                    }) + "\n")
                    kept_c += 1

        open(done_flag, "w").write(json.dumps({"groups": kept_g, "completions": kept_c}))
        n_groups += kept_g
        n_completions += kept_c
        print(f"[verify] {base}: groups={kept_g} completions={kept_c}")

    summary = {"n_groups": n_groups, "n_completions": n_completions, "n_configs": len(CONFIGS)}
    with open(os.path.join(args.out_dir, "_summary_verify.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[verify] SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
