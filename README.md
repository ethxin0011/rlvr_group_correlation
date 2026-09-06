# RLVR Group Correlation

Measures within-group verifier-error correlation on real RLVR rollouts:
whether the *k* completions GRPO samples per prompt are scored independently
by an automatic verifier, or whether their errors are correlated because
they share surface answer form. Part 2 of a two-part series; Part 1 is
[*Where the Verifier Fails*](https://arxiv.org/abs/2609.01354)
(arXiv:2609.01354).

**About**

This repository contains the full code pipeline and paper source for a research project measuring within-group verifier-error correlation in reinforcement learning with verifiable rewards (RLVR).

**Goal**
Group-based RLVR methods like GRPO sample multiple completions per prompt and score each with an automatic verifier, computing a training signal from the group's relative rewards. This project tests whether those per-completion verifier judgments are actually independent, as most existing methods assume, or whether they're correlated because completions to the same prompt tend to share answer format. The project measures this directly on real generated rollouts, breaks the correlation down by answer category, and evaluates whether a lightweight category-aware verifier selection can help.


**Data**
Final aggregated data results are uploaded and accessible here for this experiment result: https://github.com/ethxin0011/rlvr_group_correlation/tree/main/data

**Headline result:** pooled within-group ICC ρ = 0.530 (95% CI
[0.500, 0.560]) on 24,998 real k=8 rollout groups from Qwen2.5-1.5B —
equivalent to a Kish effective sample size of 1.70 out of 8 nominally
independent rollouts. Correlation is concentrated in structurally/
semantically complex answer categories (LaTeX, symbolic expressions,
intervals), not the whitespace/punctuation categories that dominate the
aggregate error rate in Part 1.

## Pipeline stages

| Stage | Purpose | Compute |
|---|---|---|
| 1 | Prompt pool + k=8 rollout generation + pilot gate | GPU + CPU |
| 2 | Rule-based verification, advantage-replay, gold-set sampling | CPU only |
| 3 | Model-based verification (CompassVerifier-3B) on a budget-capped sample | GPU |
| 4 | Join human labels + model verdicts + rule verdicts into final results | CPU only |

Run order: `01 → 02/02b → 03 (gate) → 04/05 (full generation) → 06 → 07 →
08 → 09 → 10 → 11`. Stage 1's pilot gate must pass before committing to
full-scale generation (04/05); see `src/pilot_gate.py` docstring for the
three gate conditions and why each exists.

## Folder structure

```
rlvr_group_correlation/
├── README.md                       this file
├── submit_stage1.py                 one-command job submitter
├── src/
│   ├── prompts.py                   builds the 50k prompt pool (GSM8K + MATH + DeepMath-103K)
│   ├── extract.py                   raw answer extraction + certified category taxonomy
│   ├── generate.py                  k=8 GPU rollout generation (vLLM), group-integrity enforced
│   ├── generate_cpu.py               CPU fallback rollout generation (pilot-scale only)
│   ├── pilot_gate.py                 3-gate check + preliminary rho_cat (ICC, Kish n_eff)
│   ├── rule_verify.py                4-config rule verifier pass (strict/loose/numeric/flex)
│   ├── advantage_replay.py           GRPO advantage replay: sign-flip, degeneracy vs. Jensen baseline
│   ├── gold_sampling.py              stratified sampling for human-labeled gold set
│   ├── select_compass_sample.py      budget-capped subsample for model-based verification
│   ├── compass_verify.py             CompassVerifier-3B pass (vLLM or transformers backend)
│   └── final_analysis.py             joins human labels + CompassVerifier + rule verdicts
├── jobs/                             Azure ML command job YAMLs, one per pipeline step
├── env/                              Azure ML environment specs (CPU / GPU / CPU+torch)
```

## Data outputs (not checked into git — see datastore paths in job YAMLs)

| Artifact | Produced by | Scale |
|---|---|---|
| Rollout corpus | `generate.py` (jobs 04/05) | 24,998 groups (1.5B, full) + 6,500 groups (7B-AWQ, partial/exploratory) |
| Rule verdict matrix | `rule_verify.py` (job 06) | 4 configs × full corpus |
| Advantage-replay report | `advantage_replay.py` (job 07) | degeneracy + sign-flip rates |
| Gold-labeled set | `gold_sampling.py` (job 08) + human annotation | 5,556 traces, 200 double-annotated |
| CompassVerifier verdicts | `compass_verify.py` (job 10) | 2,367 traces (budget-capped) |
| Final joined report | `final_analysis.py` (job 11) | IAA, per-category accuracy, router evaluation |

## Key methodological notes

- **Group integrity is the core design constraint.** Every rollout request asks for all *k* completions in one call; groups with fewer than *k* returned completions are dropped and counted, never backfilled (see `generate.py`).
- **ρ_cat is a marginal, not conditional, correlation** — pooled across prompts of varying difficulty. See `main.tex` §3.4 for what this does and does not establish.
- **Degeneracy-vs-Jensen-baseline gap is not attributed solely to correlated verifier error** — a single global pass rate also fails to capture between-prompt difficulty heterogeneity. See `main.tex` §3.6.
- **The 7B comparison is exploratory**, confounded by AWQ quantization vs. the 1.5B run's full precision, and based on a partial corpus (13/50 planned shards).
- **Gold-set labels are human-adjudicated** by two independent annotators; the router's category-selection rule is fit and evaluated in-sample (no held-out split) — see Limitations in `main.tex`.

```

## License / citation

Code and data released for reproducibility. If you use this work, please
cite both parts of the series
