"""
Stage 2 / step B: advantage-replay -- prices verifier disagreement in gradient
terms instead of verdict terms.

For each group and each verifier config, compute the GRPO-style group-relative
advantage a_i = (r_i - mean(r)) / (std(r) + eps). Then, per verifier PAIR,
measure:

  sign_flip_rate      fraction of completions where sign(a_i^A) != sign(a_i^B)
  group_corruption    fraction of GROUPS with >=1 sign flip between A and B
  degenerate_rate[V]  fraction of groups with std(r)==0 under verifier V
                      (zero gradient for every completion in that group)

Degeneracy is also reported as EXCESS over the Jensen/i.i.d. baseline:
predicted = p^k + (1-p)^k, where p is verifier V's GLOBAL mean reward.
This is the classical policy-stochasticity floor (Gradient Starvation in
Binary-Reward GRPO, arXiv:2605.07689) -- report only the excess above it,
never the raw degeneracy rate, or a reviewer will mistake known policy
stochasticity for a new finding.

Requires: rule_verify.py output (verdict matrix) joined against the original
rollout groups (for k and group membership). CPU only.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

VERIFIER_NAMES = ["strict", "loose", "numeric", "flex"]


def resolve_files(path_arg, suffix=".jsonl"):
    if os.path.isdir(path_arg):
        found = glob.glob(os.path.join(path_arg, "**", f"*{suffix}"), recursive=True)
    else:
        found = glob.glob(path_arg)
    return sorted(f for f in found if os.path.isfile(f))


def load_verdicts(verdict_dir, only_model=None):
    """group_id -> cid -> {verifier: 0/1/None}"""
    by_group = defaultdict(dict)
    for fp in resolve_files(verdict_dir, "_verdicts.jsonl"):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if only_model and r["model"] != only_model:
                    continue
                by_group[r["group_id"]][r["cid"]] = r["verdicts"]
    return by_group


def resolved_reward(verdicts, name):
    """Fall back to loose when numeric/flex is inapplicable (None) for a trace."""
    v = verdicts.get(name)
    if v is None:
        v = verdicts.get("loose")
    return v if v is not None else 0


def advantage(rewards):
    n = len(rewards)
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0] * n, True, mu
    return [(r - mu) / sd for r in rewards], False, mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="original rollout dir (for k, group order)")
    ap.add_argument("--verdicts", required=True, help="rule_verify.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only_model", default=None)
    args = ap.parse_args()

    print("[replay] loading verdicts...")
    verdict_map = load_verdicts(args.verdicts, args.only_model)
    print(f"[replay] {len(verdict_map)} groups have verdicts")

    n_groups = 0
    global_sum = defaultdict(float)
    global_n = defaultdict(int)
    degenerate = defaultdict(int)
    sign_flip_pairs = defaultdict(lambda: [0, 0])  # (pair) -> [flips, total_traces]
    corrupted_pairs = defaultdict(int)

    rollout_files = resolve_files(args.rollouts, ".jsonl")
    rollout_files = [f for f in rollout_files if "_verdicts" not in f]

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
                vmap = verdict_map.get(gid)
                if not vmap:
                    continue
                comps = sorted(g["completions"], key=lambda c: c["cid"])
                if not all(c["cid"] in vmap for c in comps):
                    continue

                n_groups += 1
                adv_by_verifier = {}
                deg_by_verifier = {}
                for name in VERIFIER_NAMES:
                    rewards = [resolved_reward(vmap[c["cid"]], name) for c in comps]
                    global_sum[name] += sum(rewards)
                    global_n[name] += len(rewards)
                    a, is_deg, _ = advantage(rewards)
                    adv_by_verifier[name] = a
                    deg_by_verifier[name] = is_deg
                    degenerate[name] += int(is_deg)

                for i in range(len(VERIFIER_NAMES)):
                    for j in range(i + 1, len(VERIFIER_NAMES)):
                        A, B = VERIFIER_NAMES[i], VERIFIER_NAMES[j]
                        pair = f"{A}_vs_{B}"
                        aA, aB = adv_by_verifier[A], adv_by_verifier[B]
                        flips_here = 0
                        for x, y in zip(aA, aB):
                            sign_flip_pairs[pair][1] += 1
                            if (x > 0 and y < 0) or (x < 0 and y > 0):
                                flips_here += 1
                                sign_flip_pairs[pair][0] += 1
                        if flips_here > 0:
                            corrupted_pairs[pair] += 1

    k = 8  # fixed by design (Stage 1)
    report = {"n_groups": n_groups, "k": k, "per_verifier": {}, "pairwise": {}}

    for name in VERIFIER_NAMES:
        p = global_sum[name] / max(global_n[name], 1)
        jensen_predicted = p ** k + (1 - p) ** k
        observed = degenerate[name] / max(n_groups, 1)
        report["per_verifier"][name] = {
            "global_mean_reward_p": round(p, 4),
            "degenerate_rate_observed": round(observed, 4),
            "jensen_predicted_degenerate": round(jensen_predicted, 4),
            "excess_degenerate_over_jensen": round(observed - jensen_predicted, 4),
        }

    for pair, (flips, total) in sign_flip_pairs.items():
        report["pairwise"][pair] = {
            "sign_flip_rate": round(flips / max(total, 1), 4),
            "group_corruption_rate": round(corrupted_pairs[pair] / max(n_groups, 1), 4),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
