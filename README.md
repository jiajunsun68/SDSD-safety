# SDSD-Safety

On-policy self-distillation with sampled demonstrations for LLM safety
alignment. The project adapts SDSD (Nicolicioiu et al., 2026) to safety: the
student samples multiple rollouts, a local Qwen3Guard judge selects qualified
safe rollouts as teacher demonstrations, and a frozen-reference/EMA teacher
provides token-level reverse-KL supervision on the student's own prefixes.

Progress and the live task queue are in [`status.md`](status.md). Method details
and the exact safety adaptation are in [`docs/method.md`](docs/method.md).

## Main comparison

All methods use the same backbone (`Qwen/Qwen3-4B-Instruct-2507`), training
prompts, LoRA parameterization, random seed, decoding settings, and evaluators:

| Method | Training signal |
|---|---|
| Base | no update |
| SFT | external safe/helpful responses, assistant-token cross-entropy |
| GRPO | group-normalized continuous Qwen3Guard safety margin |
| OPSA | frozen base teacher with type-conditional safety system prompts, on-policy mixed KL |
| SDSD-Safety | EMA teacher conditioned on sampled qualified safe demonstrations, on-policy reverse KL |

Reported metrics: HarmBench official ASR (official Llama-2-13B classifier),
XSTest safe/unsafe refusal rates (official string matching), and GSM8K
exact-match accuracy.

## Environment

The machine already provides a working training environment; no large wheel was
installed for this project.

```bash
PY=/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python
# Python 3.11.15, torch 2.5.1+cu121, transformers 5.8.0, peft 0.19.1, datasets 4.8.5
```

Local assets (not stored in Git):

```text
model:  /data4/sjj/models/Qwen3-4B-Instruct-2507          (ModelScope)
judge:  /data4/sjj/catelogue/alignment/models/Qwen3Guard-Gen-4B
data:   /data4/sjj/datasets/alpaca-cleaned
        /data4/sjj/datasets/xstest/xstest_prompts.csv
        /data4/sjj/alignment/harmbench_official_8e1604d
        /data4/szs/model_weights/HarmBench-Llama-2-13b-cls
```

## Reproduce

```bash
cd /data4/sjj/catelogue/sdsd_safety
PY=/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python

# 1. Fixed train/dev/test splits (Hugging Face cache + ModelScope data)
$PY -m sdsd_safety.prepare_data --config configs/qwen3_4b.yaml

# 2. Train one method (use a free GPU; 200 steps, seed 42)
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.train --config configs/qwen3_4b.yaml \
  --override train.method=sft --run-id sft_seed42
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.train --config configs/qwen3_4b.yaml \
  --override train.method=grpo --run-id grpo_seed42
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.train --config configs/qwen3_4b.yaml \
  --override train.method=opsa --run-id opsa_seed42
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.train --config configs/qwen3_4b.yaml \
  --override train.method=sdsd --run-id sdsd_self_k1_seed42

# 3. Generate benchmark outputs for a checkpoint (base uses --adapter none)
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.evaluate --phase generate \
  --run-id base --adapter none --benchmarks harmbench,xstest,gsm8k
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.evaluate --phase generate \
  --run-id sdsd_self_k1_seed42 --adapter runs/sdsd_self_k1_seed42/checkpoint

# 4. Score all saved HarmBench generations once with the official classifier
CUDA_VISIBLE_DEVICES=0 $PY -m sdsd_safety.evaluate --phase judge

# 5. Merge run summaries
$PY -m sdsd_safety.evaluate --phase summarize
```

`results/summary.csv` is the machine-readable main table. `runs/<run_id>/`
contains the adapter-only checkpoint, `train.log`, `metrics.jsonl`,
`dev_metrics.json` and `run.json` (git commit, command, config).

## Design choices recorded before the main runs

- `Qwen/Qwen3-4B-Instruct-2507` is a 4B-class non-thinking Qwen3 model available
  on ModelScope; it keeps rollout cost low and avoids thinking-token inflation.
  The 8B scale-up is left for a follow-up, not mixed into the main table.
- Harmful prompts: PKU-SafeRLHF rows with exactly one safe response. A
  Qwen3Guard audit of the external responses keeps 4,966/8,512 (58.3%).
- Normal prompts: alpaca-cleaned instructions with non-refusal responses.
- Main SDSD demonstrations: for harmful prompts only qualified **refusals** are
  used as demonstrations (external/fixed refusal fallback); normal prompts use
  qualified non-refusal answers. Allowing any safe answer as a demo collapsed
  to a repeated-output policy, so it is excluded from the main table.
- Main SFT: refusal targets (442 external refusals + 2,758 fixed brief
  refusals), because external safe answers are often safe alternatives rather
  than refusals. The external-answer SFT is kept as an exploratory failure.
- Rollouts: 4 per prompt, temperature 1.0, top-p 1.0, max 128 new tokens.
- LoRA r=32, alpha=64, dropout=0.05 on all attention/MLP projections; AdamW,
  grad clip 1.0, 16 prompts per step, 200 steps, seed 42. SFT/GRPO/OPSA use
  lr 1e-4; the stable SDSD run uses lr 5e-5, reference-KL 0.03, EMA teacher
  update rate 0.02 (the initial lr-1e-4/EMA-0.05 run degenerated).
- SDSD teacher context is the original question, one sampled demonstration,
  and the student prefix. K=3 averages three independent per-demo losses.
- OPSA: frozen base teacher, official type-conditional prompts, mixed KL
  (0.5 forward + 0.5 reverse).
- Ablations use 100 steps (budget); this is a stated confound vs the 200-step
  main runs.
- Checkpoint policy: adapter-only final checkpoint per run; no per-step or
  full-model checkpoints.

## Results

Main comparison (official HarmBench classifier; XSTest is the Qwen3Guard
semantic refusal rate, with official string-match in
`results/tables/main.csv`; 200 steps, seed 42):

| Method | HarmBench ASR↓ | XSTest safe refusal↓ | XSTest unsafe refusal↑ | GSM8K↑ |
|---|---|---|---|---|
| Base | 4.75 [2.75, 7.00] | 7.6 | 82.0 | 90.5 |
| Safety SFT (refusal targets) | 2.00 [0.75, 3.50] | 92.8 | 100.0 | 87.0 |
| Safety GRPO (margin + refusal) | 0.00 [0.00, 0.00] | 98.0 | 100.0 | 80.5 |
| OPSA | 5.25 [3.25, 7.50] | 35.2 | 95.0 | 91.5 |
| SDSD-Safety (K=1, self refusal demos) | **0.75 [0.00, 1.75]** | 73.2 | 100.0 | 70.5 |

Ablations (100 steps, so not budget-matched to the 200-step main runs):

| Variant | HarmBench ASR↓ | XSTest safe refusal↓ | XSTest unsafe refusal↑ | GSM8K↑ |
|---|---|---|---|---|
| SDSD, external refusal demos, K=1 | 5.25 | 17.6 | 94.0 | 87.5 |
| SDSD, self refusal demos, K=3 | 12.00 | 25.6 | 98.5 | 90.0 |
| SFT on external safe answers (exploratory) | 28.50 | 6.8 | 59.0 | 71.0 |
| SDSD, any-safe demos (200 steps) | invalid: repeated-output degeneration | — | — | — |

Key findings:

1. **SDSD gives the largest safety gain**: HarmBench ASR falls from 4.75% to
   0.75% (non-overlapping 95% CI), and all 200 XSTest unsafe prompts are
   refused. The cost is a 73.2% over-refusal rate on XSTest safe prompts and a
   20-point GSM8K drop (90.5 -> 70.5). The latter is partly a teacher-context
   artifact: the model prepends "Of course — here is your complete ... answer"
   to GSM8K responses.
2. **OPSA preserves capability but not HarmBench safety**: GSM8K reaches 91.5%
   and over-refusal is only 35.2%, but ASR is 5.25%, statistically
   indistinguishable from Base.
3. **SFT and GRPO are safe but over-refuse**: ASR 2.0%/0.0%, but XSTest safe
   refusal 92.8%/97.6% and GSM8K 87.0%/80.5%.
4. **Refusal demos are essential**: using any safe rollout (including safe
   alternative answers) as the demo collapses to a repeated "I have a human."
   output; external refusal demos and K=3 preserve capability but lose most of
   the ASR gain at the 100-step budget.
5. **Mechanism**: SDSD produces the largest demo sensitivity (prefix log-prob
   delta +4.66), the largest KL to base (8.39) and concentrates 72% of prefix
   KL in the early half of the refusal prefix. OPSA makes a tiny,
   early-concentrated update (KL 0.355, early share 0.74), consistent with its
   capability preservation. The full table is in
   [`results/mechanism_main.json`](results/mechanism_main.json).

Raw generations, per-run logs and adapter checkpoints stay outside Git
(`results/raw/`, `runs/`); machine-readable summaries are in
[`results/summary.csv`](results/summary.csv),
[`results/tables/main.csv`](results/tables/main.csv) and
[`results/tables/ablations.csv`](results/tables/ablations.csv).

## Attribution

- SDSD paper: arXiv:2606.26091 (algorithm and mechanism analysis).
- OPSA paper/code: arXiv:2605.15239, github.com/FYYFU/OPSA (safety prompts and
  mixed-KL default; Apache-2.0).
- HarmBench official behavior set, prompts and Llama-2-13B classifier.
- XSTest refusal string matching: github.com/paul-rottger/xstest (CC-BY-4.0).
- Base model and alpaca-cleaned data via ModelScope/Hugging Face.
