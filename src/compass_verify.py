"""
Stage 3 / step B: run CompassVerifier-3B over the budget-capped sample.

WHAT THIS IS FOR:
Rule verifiers (strict/loose/numeric/flex) can only ever disagree in ways a
human anticipated when writing string-matching rules. CompassVerifier-3B is a
model-based verifier trained on ~1M human-labeled verdicts, so it disagrees
with rule verifiers for GENUINELY DIFFERENT reasons (semantic equivalence,
partial credit, malformed-but-correct reasoning). Running it on the
disagreement-heavy sample gives you a second, independent verifier opinion --
the actual "inter-verifier disagreement" the paper is about, rather than the
strict-vs-flex PROXY used in the gate. Runs on the T4 (16GB); CompassVerifier-
3B in fp16 is ~6GB, comfortable with a KV cache.

MODEL NOTE: verify the exact HF repo id before running --
"opencompass/CompassVerifier-3B" is used below based on public documentation;
if the org/repo path has changed, update --model accordingly. If vLLM 0.6.3
cannot load it (architecture too new, same failure mode as Qwen3), fall back
to plain transformers generation (slower, but works with any transformers
version -- see the --backend flag).

Judgment scheme follows CompassVerifier's own protocol: A=correct,
B=incorrect, C=quality problem / cannot judge.
"""

import argparse
import csv
import json
import os
import re

PROMPT_TEMPLATE = """You are an expert verifier for mathematical answers. Given a question, a reference (gold) answer, and a candidate final answer, judge whether the candidate answer is mathematically equivalent to the reference answer.

Question: {question}

Reference Answer: {gt_answer}

Candidate Answer: {raw_answer}

Respond with exactly one letter:
A = the candidate answer is correct (mathematically equivalent to the reference)
B = the candidate answer is incorrect
C = cannot be judged (malformed, missing, or a quality problem with the question/reference itself)

Judgment:"""


def parse_judgment(text):
    m = re.search(r"\b([ABC])\b", text.strip()[:20])
    return m.group(1) if m else None


def run_vllm(rows, model, max_model_len, gpu_mem_util):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=gpu_mem_util,
              max_model_len=max_model_len, trust_remote_code=True, seed=20260904)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    prompts = [PROMPT_TEMPLATE.format(**r) for r in rows]
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outputs]


def run_transformers(rows, model, max_model_len):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16,
                                             trust_remote_code=True, device_map="cuda")
    m.eval()
    texts = []
    for r in rows:
        prompt = PROMPT_TEMPLATE.format(**r)
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_model_len).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, do_sample=False, max_new_tokens=8,
                             pad_token_id=tok.eos_token_id)
        texts.append(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="opencompass/CompassVerifier-3B")
    ap.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--max_model_len", type=int, default=1024)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--batch_size", type=int, default=200,
                    help="write progress every N rows (resumability)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.sample_csv, encoding="utf-8")))
    print(f"[compass] {len(rows)} traces to verify, backend={args.backend}")

    done_ids = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            done_ids.add(json.loads(line)["group_id"] + "::" + str(json.loads(line)["cid"]))
    remaining = [r for r in rows if f"{r['group_id']}::{r['cid']}" not in done_ids]
    print(f"[compass] {len(rows) - len(remaining)} already done, {len(remaining)} remaining")

    with open(args.out, "a", encoding="utf-8") as fout:
        for i in range(0, len(remaining), args.batch_size):
            batch = remaining[i:i + args.batch_size]
            if args.backend == "vllm":
                texts = run_vllm(batch, args.model, args.max_model_len, args.gpu_mem_util)
            else:
                texts = run_transformers(batch, args.model, args.max_model_len)
            for r, text in zip(batch, texts):
                fout.write(json.dumps({
                    "group_id": r["group_id"], "cid": int(r["cid"]),
                    "primary_cat": r["primary_cat"],
                    "agreement_pattern": r["agreement_pattern"],
                    "compass_judgment": parse_judgment(text),
                    "compass_raw": text.strip()[:50],
                }) + "\n")
            fout.flush()
            print(f"[compass] {min(i+args.batch_size, len(remaining))}/{len(remaining)} done",
                  flush=True)

    print(f"[compass] wrote -> {args.out}")


if __name__ == "__main__":
    main()
