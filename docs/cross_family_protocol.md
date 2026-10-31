# Cross-family confirmation protocol (pre-registered 2026-10-31)

## Model choice

The prompt prefers Llama-3.1-8B-Instruct for cross-family confirmation. The local
Llama directory `/data2/sjr/models-mirror/Meta-Llama-3.1-8B-Instruct` contains
`WEIGHTS_REMOVED.json` (weights intentionally deleted on 2026-10-26), so that
checkpoint cannot be loaded and no download is assumed available. The
pre-registered fallback is used: **Qwen2.5-7B-Instruct** at
`/data4/sjj/catelogue/alignment/models/Qwen2.5-7B-Instruct` (complete local
safetensors; different model family/tokenizer from Qwen3-4B-Instruct-2507).
The fallback and its reason are recorded here and in every run.json.

## Training configurations

All runs start from the same Qwen2.5-7B-Instruct Base, use the audited
`data/train.jsonl`, 100 steps, seed 42, LoRA r=32/alpha=64 on
q/k/v/o/gate/up/down, lr 5e-5, `ref_kl_coef=0.03`, EMA teacher rate 0.02,
one self-sampled harmful demonstration, and normal demos from the audited
external response.

| run_id | role | frozen_teacher_weight | frozen_teacher_demo |
|---|---|---:|---|
| `sdsd_v3_qwen25_anchor_100_seed42` | original demo-anchor | 0.25 | true |
| `sdsd_v3_qwen25_plainfrozen_100_seed42` | mechanism-derived repair | 0.25 | false |

These are the original method and its best current repair. A K1 control is added
only if the anchor run shows an attack-specific effect worth decomposing.

## Evaluation

The same standard protocols as the main study are used: HarmBench standard-200
greedy, XSTest (string refusal primary; Qwen3Guard semantic supplementary), and
GSM8K with the lm-evaluation-harness flexible-extract primary scorer. Each model
uses its own tokenizer's chat template and EOS token (`<|im_end|>` for Qwen2.5).
Base Qwen2.5 is evaluated without training (`base_qwen25_v1`).

Fixed-attack evaluation uses the same already-frozen HumanJailbreaks index 6
protocol on HarmBench text-test standard behaviors (159 x 16 samples), so the
cross-family attack result is directly comparable to the Qwen3 result.

## Independent attack family (frozen before outcomes)

The original prompt index 6 was selected on 41 validation behaviors and confined
on 159 test behaviors. For independent confirmation we pre-register a *different*
official HumanJailbreaks prompt: **index 8** (`@Clyde ignore all previous
instructions`, instruction-override family; prompt sha256 prefix
`9aed3861d2d0`, 318 Qwen3 tokens). Index 8 is outside the 0-7 dev-sweep range and
was not selected using any attack outcome; it was chosen because it instantiates
a different attack mechanism (instruction override rather than character
role-play). Rationale and hash are recorded before generation.

Protocol: `sdsd-safety-hb-humanjb-idx8-test159-indep-v1`; HarmBench text-test
standard-159 behaviors x 16 samples; temperature 1, top-p 1, top-k 0, max 320 new
tokens; official HarmBench Llama-2-13B classifier reading only the original
behavior; same ASR@1/4/16 estimator and paired bootstrap. Models: Base, original
anchor, plain-frozen repair (R1), and the R2 branch if available. This remains a
fixed-prompt protocol, not an adaptive or transfer attack; it tests whether the
anchor vulnerability and its repair replicate under a second attack family.

## Reporting rule

A cross-family negative (no anchor vulnerability) is a valid result and does not
supersede the Qwen3 finding; it bounds generality. Results are only superseded
for confirmed implementation, scoring, or protocol errors, with the reason and
old artifact retained.
