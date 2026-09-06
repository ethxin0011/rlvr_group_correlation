"""
Stage 1 / step A: build the 50k prompt pool with intact ground-truth answers.

IMPORTANT SCALING NOTE
----------------------
MATH train (7.5k) + GSM8K train (7.47k) = ~15k prompts only.
That is NOT enough for 50k prompts x k=8 = 400k rollouts.
We top up with DeepMath-103K, which is verified and RLVR-oriented.

Composition (target 50,000 prompts):
    GSM8K train        7,473
    MATH train         7,500
    DeepMath-103K     35,027
    ----------------------------
    total             50,000

Every row carries: prompt_id, source, question, gt_answer.
gt_answer is the *reference string* the verifiers will compare against.
"""

import argparse
import hashlib
import json
import os
import random
import re

from datasets import load_dataset

SEED = 20260904


def _hid(source: str, question: str) -> str:
    h = hashlib.sha1(f"{source}||{question}".encode("utf-8")).hexdigest()
    return f"{source}-{h[:16]}"


def _gsm8k_answer(ans: str) -> str:
    # GSM8K answers end with "#### <number>"
    m = re.search(r"####\s*(.+)\s*$", ans.strip())
    return m.group(1).strip().replace(",", "") if m else ans.strip()


def _math_answer(sol: str) -> str:
    # pull the last \boxed{...} with brace matching
    idx = sol.rfind(r"\boxed")
    if idx == -1:
        return ""
    i = sol.find("{", idx)
    if i == -1:
        return ""
    depth, j = 0, i
    while j < len(sol):
        if sol[j] == "{":
            depth += 1
        elif sol[j] == "}":
            depth -= 1
            if depth == 0:
                return sol[i + 1 : j].strip()
        j += 1
    return ""


def build(out_path: str, n_target: int = 50000, deepmath_n: int = 35027):
    rng = random.Random(SEED)
    rows = []

    # ---- GSM8K train ----
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        a = _gsm8k_answer(r["answer"])
        if a:
            rows.append(
                {
                    "prompt_id": _hid("gsm8k", r["question"]),
                    "source": "gsm8k",
                    "question": r["question"].strip(),
                    "gt_answer": a,
                }
            )

    # ---- MATH train (Hendrycks) ----
    math_ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for r in ds:
            a = _math_answer(r["solution"])
            if a:
                rows.append(
                    {
                        "prompt_id": _hid("math", r["problem"]),
                        "source": "math",
                        "question": r["problem"].strip(),
                        "gt_answer": a,
                    }
                )

    # ---- DeepMath-103K top-up ----
    dm = load_dataset("zwhe99/DeepMath-103K", split="train")
    dm_rows = []
    for r in dm:
        q = r.get("question") or r.get("problem")
        a = (r.get("final_answer") or "").strip()
        if q and a:
            dm_rows.append(
                {
                    "prompt_id": _hid("deepmath", q),
                    "source": "deepmath",
                    "question": q.strip(),
                    "gt_answer": a,
                }
            )
    rng.shuffle(dm_rows)
    rows.extend(dm_rows[:deepmath_n])

    # ---- dedup + trim ----
    seen, final = set(), []
    for r in rows:
        if r["prompt_id"] in seen:
            continue
        seen.add(r["prompt_id"])
        final.append(r)
    rng.shuffle(final)
    final = final[:n_target]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    print(f"[prompts] wrote {len(final)} -> {out_path}")
    print("[prompts] by source:", Counter(r["source"] for r in final))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--deepmath_n", type=int, default=35027)
    a = p.parse_args()
    build(a.out, a.n, a.deepmath_n)
