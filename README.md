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
| GRPO | group-normalized binary safety reward from Qwen3Guard |
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

# 2. Train one method (use a free GPU; 250 steps, seed 42)
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
- Harmful prompts: PKU-SafeRLHF rows with exactly one safe response. The safe
  response is the external demonstration and the SFT target.
- Normal prompts: alpaca-cleaned instructions with non-refusal responses.
- External demonstrations are identical for SFT and for SDSD's no-qualified-demo
  fallback, so the comparison does not silently change the data.
- Rollouts: 4 per prompt, temperature 1.0, top-p 1.0, max 192 new tokens.
- LoRA r=32, alpha=64, dropout=0.05 on all attention/MLP projections, AdamW,
  lr 5e-5, 250 steps, 16 prompts per step, seed 42.
- SDSD teacher: LoRA-parameter EMA with update rate 0.05; teacher context is the
  original question, one sampled qualified demonstration, and then the student
  prefix. K=3 averages three per-demonstration losses.
- OPSA: frozen base teacher, official type-conditional prompts, mixed KL
  (0.5 forward + 0.5 reverse).
- Checkpoint policy: adapter-only final checkpoint per run; no per-step or
  full-model checkpoints.

## Results

See [`results/summary.csv`](results/summary.csv) and the table in
[`status.md`](status.md). Empty cells mean the run has not completed; no
estimated values are reported.

## Attribution

- SDSD paper: arXiv:2606.26091 (algorithm and mechanism analysis).
- OPSA paper/code: arXiv:2605.15239, github.com/FYYFU/OPSA (safety prompts and
  mixed-KL default; Apache-2.0).
- HarmBench official behavior set, prompts and Llama-2-13B classifier.
- XSTest refusal string matching: github.com/paul-rottger/xstest (CC-BY-4.0).
- Base model and alpaca-cleaned data via ModelScope/Hugging Face.
