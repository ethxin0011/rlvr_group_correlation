"""
Stage 1 / step C: k=8 group-preserving rollout generation with vLLM on one T4.

GROUP INTEGRITY IS THE CONTRIBUTION.
 * one vLLM request per prompt with n=k  -> all k completions come back together
 * one JSONL row per GROUP, never per trace
 * groups with < k completions are DROPPED and counted, never backfilled

T4 (sm75) constraints, enforced below:
 * dtype MUST be float16 (no bf16 on Turing)
 * no FlashAttention-2; vLLM falls back to xformers automatically
 * Qwen3-1.7B fp16 ~3.5 GB -> comfortable
 * Qwen3-8B fp16 ~16.4 GB -> DOES NOT FIT. Use an AWQ int4 build (~5.5 GB).
   Policy quantisation is acceptable (the policy is the *subject*, not the
   measuring instrument) but MUST be disclosed in the paper's §Setup.

Resumability: output is sharded; a completed shard is skipped on restart.
"""

import argparse
import gc
import json
import os
import time

from extract import categorise, extract_final_answer

SYSTEM = (
    "You are a careful mathematical reasoner. Think step by step, then give the "
    "final answer inside \\boxed{}."
)


def load_prompts(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_chat(tok, question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True, help="short id stored in each row")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=640)
    ap.add_argument("--max_model_len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=2000, help="prompts per shard")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--quantization", default=None, help="awq / gptq / None")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args.prompts, args.limit)
    prompts = prompts[args.start :]
    print(f"[gen] {len(prompts)} prompts | model={args.model_tag} | k={args.k}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="float16",                 # MANDATORY on T4
        quantization=args.quantization,  # "awq" for the 8B build
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
        enforce_eager=False,
        swap_space=4,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    n_shards = (len(prompts) + args.shard_size - 1) // args.shard_size
    total_groups = total_dropped = 0
    t_start = time.time()

    for si in range(n_shards):
        shard_path = os.path.join(args.out_dir, f"{args.model_tag}_shard{si:04d}.jsonl")
        done_flag = shard_path + ".done"
        if os.path.exists(done_flag):
            print(f"[gen] shard {si} already complete, skipping")
            continue

        chunk = prompts[si * args.shard_size : (si + 1) * args.shard_size]
        texts = [build_chat(tok, r["question"]) for r in chunk]

        t0 = time.time()
        outputs = llm.generate(texts, sp)
        dt = time.time() - t0

        kept = dropped = 0
        with open(shard_path, "w", encoding="utf-8") as f:
            for row, out in zip(chunk, outputs):
                comps = [o.text for o in out.outputs]
                if len(comps) != args.k:
                    dropped += 1
                    continue

                completions = []
                for ci, text in enumerate(comps):
                    raw, mode = extract_final_answer(text)
                    cats, primary = categorise(raw, text)
                    completions.append(
                        {
                            "cid": ci,
                            "text": text,
                            "raw_answer": raw,          # NEVER normalised
                            "extract_mode": mode,
                            "categories": cats,
                            "primary_cat": primary,
                            "n_tokens": len(out.outputs[ci].token_ids),
                        }
                    )

                f.write(
                    json.dumps(
                        {
                            "group_id": f"{args.model_tag}::{row['prompt_id']}",
                            "prompt_id": row["prompt_id"],
                            "source": row["source"],
                            "question": row["question"],
                            "gt_answer": row["gt_answer"],
                            "model": args.model_tag,
                            "k": args.k,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "seed": args.seed,
                            "completions": completions,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1

        open(done_flag, "w").write(
            json.dumps({"kept": kept, "dropped": dropped, "seconds": dt})
        )
        total_groups += kept
        total_dropped += dropped
        rate = kept / max(dt, 1e-6)
        eta = (len(prompts) - (si + 1) * args.shard_size) / max(rate, 1e-6) / 3600
        print(
            f"[gen] shard {si+1}/{n_shards} kept={kept} dropped={dropped} "
            f"{dt:.0f}s ({rate:.2f} grp/s) ETA {eta:.1f}h",
            flush=True,
        )

    summary = {
        "model_tag": args.model_tag,
        "groups": total_groups,
        "dropped": total_dropped,
        "drop_rate": total_dropped / max(total_groups + total_dropped, 1),
        "traces": total_groups * args.k,
        "wall_hours": (time.time() - t_start) / 3600,
    }
    with open(os.path.join(args.out_dir, f"_summary_{args.model_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[gen] SUMMARY", json.dumps(summary, indent=2))

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
