"""
Stage 1 / step C-alt: k=8 group-preserving rollout generation on CPU.

Use this ONLY for the pilot gate when the T4 is unavailable. It produces the
exact same group-per-row JSONL schema as generate.py, so pilot_gate.py consumes
it unchanged.

Why this is acceptable for the gate:
  * the gate measures group STRUCTURE (mixed-group rate, category diversity,
    verifier disagreement), not model quality
  * HF `generate` with num_return_sequences=k returns all k completions from a
    single call, so group integrity is preserved exactly as in the vLLM path

Why it is NOT acceptable for the full 400k corpus:
  * throughput is roughly two orders of magnitude below the T4
  * run jobs 04/05 on the GPU once it frees

Threading: set OMP_NUM_THREADS / MKL_NUM_THREADS to the box's physical cores
(48 on E48s_v3, 16 on D16ads_v5) BEFORE torch is imported. Done below.
"""

import argparse
import json
import os
import time

# must precede torch import
_T = os.environ.get("N_THREADS", "")
if _T:
    os.environ.setdefault("OMP_NUM_THREADS", _T)
    os.environ.setdefault("MKL_NUM_THREADS", _T)

from extract import categorise, extract_final_answer  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model_tag", default="qwen2.5-1.5b-cpu-pilot")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch_prompts", type=int, default=2,
                    help="prompts per forward batch; effective batch = this * k")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=50)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"[cpu-gen] torch threads = {torch.get_num_threads()}", flush=True)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args.prompts, args.limit)
    print(f"[cpu-gen] {len(prompts)} prompts | model={args.model} | k={args.k}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,   # fp32 is fastest on CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    def chat(q):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    n_shards = (len(prompts) + args.shard_size - 1) // args.shard_size
    total_groups = total_dropped = 0
    t_start = time.time()

    for si in range(n_shards):
        shard_path = os.path.join(args.out_dir,
                                  f"{args.model_tag}_shard{si:04d}.jsonl")
        done_flag = shard_path + ".done"
        if os.path.exists(done_flag):
            print(f"[cpu-gen] shard {si} done, skipping", flush=True)
            continue

        chunk = prompts[si * args.shard_size:(si + 1) * args.shard_size]
        kept = dropped = 0
        t0 = time.time()

        with open(shard_path, "w", encoding="utf-8") as fout:
            for bi in range(0, len(chunk), args.batch_prompts):
                batch = chunk[bi:bi + args.batch_prompts]
                texts = [chat(r["question"]) for r in batch]
                enc = tok(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024)

                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_tokens,
                        num_return_sequences=args.k,   # <-- group integrity
                        pad_token_id=tok.pad_token_id,
                    )

                in_len = enc["input_ids"].shape[1]
                gen = out[:, in_len:]
                # rows are ordered prompt-major: [p0 x k, p1 x k, ...]
                for pi, row in enumerate(batch):
                    seqs = gen[pi * args.k:(pi + 1) * args.k]
                    if seqs.shape[0] != args.k:
                        dropped += 1
                        continue
                    completions = []
                    for ci in range(args.k):
                        ids = seqs[ci]
                        text = tok.decode(ids, skip_special_tokens=True)
                        raw, mode = extract_final_answer(text)
                        cats, primary = categorise(raw, text)
                        completions.append({
                            "cid": ci,
                            "text": text,
                            "raw_answer": raw,
                            "extract_mode": mode,
                            "categories": cats,
                            "primary_cat": primary,
                            "n_tokens": int((ids != tok.pad_token_id).sum()),
                        })
                    fout.write(json.dumps({
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
                        "backend": "cpu-transformers",
                        "completions": completions,
                    }, ensure_ascii=False) + "\n")
                    kept += 1

                el = time.time() - t0
                done_here = bi + len(batch)
                rate = done_here / max(el, 1e-6)
                print(f"[cpu-gen] shard {si} {done_here}/{len(chunk)} "
                      f"{el:.0f}s ({rate*60:.1f} grp/min)", flush=True)

        open(done_flag, "w").write(json.dumps({"kept": kept, "dropped": dropped}))
        total_groups += kept
        total_dropped += dropped
        print(f"[cpu-gen] shard {si+1}/{n_shards} kept={kept} dropped={dropped}",
              flush=True)

    summary = {
        "model_tag": args.model_tag,
        "backend": "cpu-transformers",
        "groups": total_groups,
        "dropped": total_dropped,
        "traces": total_groups * args.k,
        "wall_hours": (time.time() - t_start) / 3600,
    }
    with open(os.path.join(args.out_dir, f"_summary_{args.model_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[cpu-gen] SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
