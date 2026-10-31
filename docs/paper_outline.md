# Paper outline: Demonstration Context Is the Vulnerability

Status: working outline for an ACL-style submission. All numbers below are from
actual local runs; compact artifacts are under `results/`, raw generations and
checkpoints are intentionally not committed.

## Working title

**Demonstration Context Is the Vulnerability: Role-Play Jailbreaks in
Demonstration-Conditioned Safety Distillation**

Alternatives: "The Frozen Teacher Is Not Enough: Context-Specific Jailbreak
Risk in On-Policy Safety Distillation"; "Hidden in the Context: Why a
Demo-Anchored Safety Distillation Method Fails Under Role-Play Attacks".

## Abstract draft

Demonstration-conditioned on-policy self-distillation (SDSD) with a frozen
teacher anchor can reduce over-refusal while retaining reasoning capability, and
on direct harmful requests it appears safe: all compared methods sit at the
HarmBench floor (0.5-3.0% greedy ASR). Using a fixed, pre-registered role-play
jailbreak prompt on held-out HarmBench behaviors, we find that the demo-anchored
variant is severely vulnerable: ASR@16 is 57.2%, versus 2.5% for the base model
and 3.1% after a minimal repair. The failure is context-specific rather than a
global loss of safety: plain harmful dev responses remain safe, capability and
over-refusal are preserved, and a direct-request judge does not expose it. A
zero-training LoRA-scale probe shows a dose threshold (anchor at s=0.25 matches
Base, s=1 explodes), and teacher-forced probes show the EMA teachers do not
prefer harmful continuations at clean starts; a short compliant prefix raises
harmful likelihood for all teachers. Removing the demonstration from the frozen
teacher branch (plain-frozen) repairs the vulnerability to Base level while
retaining flexible GSM8K and over-refusal. Increasing plain-prompt reference KL
by 10x does **not** repair it (ASR@16 59.8%), so in-distribution Base retention
is insufficient. The vulnerability does not replicate under an
instruction-override attack family (0% for Base, anchor and repair), and it
cannot be tested in Qwen2.5-7B because that Base is already at ceiling
(ASR@16 100%); there, training worsens direct-request ASR and yields no
over-refusal benefit. We report these boundaries rather than claiming general
safety improvement.

## Contributions

1. **Hidden failure in a safety method.** Direct-request ASR is at the floor,
   but a held-out fixed role-play attack exposes a 54.7 pp increase over Base
   (request-paired 95% CI [+47.2, +62.3]).
2. **Mechanism decomposition.** Dose-response scaling, teacher-forced
   context/prefix probes, and two controlled retrainings separate update
   magnitude, teacher parameters, and demonstration context. The repair is
   context-specific: removing the demo from the frozen branch works; stronger
   plain-prompt KL does not.
3. **Minimal repair with retained benefits.** Plain-frozen reaches
   ASR@1/4/16 = 1.26/2.31/3.14% (indistinguishable from Base @16), with
   XSTest safe refusal 5.6% string / 7.2% semantic and GSM8K lm-eval 90.5%
   (Base 5.6/7.6, 91.0).
4. **Boundary conditions and negative results.** No replication under an
   instruction-override family; cross-family Qwen2.5 is ceiling-limited and
   shows a different, unfavorable trade-off. These results bound the claim and
   motivate context-aware evaluation of safety distillation.

## Method summary

- **Base method (anchor):** on-policy student rollouts, one self-sampled safe
  demonstration for harmful requests, teacher distribution
  `0.75 q_EMA,demo + 0.25 q_Base,demo`, plus reference KL to the adapter-disabled
  base on plain prompts.
- **Plain-frozen (R1):** same mixture, but the frozen teacher sees only the
  original request (`frozen_teacher_demo=false`); the EMA teacher still sees the
  demonstration.
- **R2 control:** K1 SDSD (`frozen_teacher_weight=0`) with
  `ref_kl_coef=0.3` instead of 0.03; tests whether stronger in-distribution Base
  retention substitutes for the context fix.
- All retrainings start from the same Base, use the same audited data, 100
  steps, seed 42, LoRA r=32/alpha=64, lr 5e-5.

## Experimental setup

- **Model:** Qwen3-4B-Instruct-2507 (main); Qwen2.5-7B-Instruct (cross-family
  fallback because local Llama-3.1-8B weights were removed).
- **Data:** audited 2,000 harmful + 2,000 normal training requests; 200/200 dev
  and test splits; Qwen3Guard labels; no test leakage.
- **Direct benchmarks:** HarmBench standard-200 greedy; XSTest string and
  Qwen3Guard semantic refusal; GSM8K with the public lm-evaluation-harness
  flexible-extract primary scorer (exact retained as a format diagnostic).
- **Fixed attack:** official HumanJailbreaks prompt index 6, selected on 41
  validation behaviors by a pre-registered rule, then evaluated on 159 held-out
  text-test standard behaviors with 16 samples (temperature 1, top-p 1, top-k 0,
  max 320); official Llama-2-13B HarmBench classifier reads only the original
  behavior. ASR@k uses the combinatorial estimator and request-level paired
  bootstrap (10,000 reps); unknown labels are reported, never treated as safe.
- **Independent attack:** pre-registered HumanJailbreaks index 8
  (instruction override; outside the selection range), same 159 x 16 protocol.
- **Mechanism probes:** zero-training LoRA scaling (`W=W_base+s dW` in memory,
  s in {0, 0.25, 0.5, 1}); teacher-forced mean token log-probability and
  first-token KL for Base / EMA-K1 / EMA-anchor x plain/demo context x
  no/short-compliant prefix.

## Main results (files)

| Result | Artifact |
|---|---|
| Seven-run fixed-attack ASR and paired differences | `results/tables/attack_humanjb_idx6_test159.md`, `results/analysis/attack_humanjb_idx6_test159.json` |
| Repair / R2 paired comparisons | `results/analysis/repair_paired_idx6_test159.json` |
| P2 scale pilot | `results/tables/scale_pilot.md`, `results/analysis/scale_pilot.json` |
| P1 teacher-context probe | `results/tables/teacher_context_probe.md`, `results/analysis/teacher_context_probe.json` |
| Independent idx8 attack | `results/tables/attack_humanjb_idx8_test159.md` (raw only; compact table from analyze) |
| Cross-family Qwen2.5 | `results/tables/cross_family_qwen25.md`, `results/analysis/cross_family_qwen25.json`, `results/tables/attack_humanjb_idx6_test159_qwen25.md` |
| Standard benchmarks and scorer metadata | `results/tables/main.md`, `results/summary.json`, `results/raw/*/*_summary.json` |
| Protocol definitions | `docs/asr_protocol.md`, `docs/gsm8k_protocol.md`, `docs/cross_family_protocol.md` |

Key Qwen3 attack table (ASR@1/4/16 %):

| Run | @1 | @4 | @16 |
|---|---:|---:|---:|
| Base | 0.94 | 1.50 | 2.52 |
| OPSA | 2.75 | 5.29 | 10.69 |
| SDSD K1 | 7.90 | 12.04 | 15.09 |
| external-demo K1 | 20.83 | 28.90 | 32.70 |
| anchor | 37.89 | 49.91 | 57.23 |
| R1 plain-frozen | 1.26 | 2.31 | 3.14 |
| R2 K1 + ref KL 0.3 | 43.67 | 54.08 | 59.75 |

## Analysis narrative

1. Open with the floor effect on direct requests and the held-out attack
   (Table above). Emphasize that the attack prompt was selected only on
   validation behaviors and frozen before test evaluation.
2. Show P2 dose-response: the failure requires a sufficiently large update.
   In the 16-sample promotion, anchor s=0.25 is indistinguishable from Base at
   ASR@16 (+2.44 pp, CI [0.00,+7.32]), s=0.5 is elevated (+12.20 pp,
   [+2.44,+24.39]), and s=1 reaches 48.78% (+46.34 pp); normal-dev refusal is
   already 0% at s=0.25. The repair is not merely "train less": scaling is a
   diagnostic, not a deployed method (`results/tables/scale_pilot_16.md`).
3. Show P1 fixed-state probe: EMA teachers are not more harmful at clean
   starts; a compliant prefix raises harmful logP for all; normal first-token
   KL is large. Frame as descriptive conditional evidence, not a causal rollout
   estimate.
4. Compare R1 and R2. R1 removes the demo context from the frozen branch and
   restores Base-level attack robustness; R2 keeps the strong plain-prompt KL
   and remains as vulnerable as anchor. This rules out "just regularize harder
   toward Base" as a sufficient explanation.
5. Report boundaries: idx8 zero; Qwen2.5 ceiling and unfavorable direct-ASR
   trade-off; XSTest string scorer invalid for Qwen2.5, semantic reported.
6. Conclude with an evaluation recommendation: safety distillation papers
   should report context-shifted fixed attacks, not only direct requests, and
   should ablate which branch sees the demonstration.

## Limitations

- Main mechanism evidence is one model family, one seed, and one attack family
  (role-play). The independent family was a valid negative, not a replication.
- The attack is a fixed prompt, not adaptive or transfer; ASR should not be read
  as an upper bound.
- LoRA scaling is a descriptive zero-training probe; it does not establish that
  a deployable low-rank solution exists.
- Cross-family evidence is ceiling-limited on the attack; Qwen2.5 results show
  an unfavorable direct-ASR trade-off and no over-refusal benefit.
- XSTest string refusal is brittle for Qwen2.5 fine-tunes; semantic refusal is
  reported with an explicit scorer caveat.
- No multi-seed confidence intervals for XSTest/GSM8K; attack CIs are
  request-paired but from a single training seed.
- The HarmBench classifier and Qwen3Guard judge have their own errors; unknown
  labels are reported and bounded, not treated as safe.

## Claim-evidence matrix

| Claim | Evidence | Status |
|---|---|---|
| Anchor is severely vulnerable under held-out role-play attack | idx6 test159 x16, 57.23% vs Base 2.52%, paired CI | Supported |
| Failure is invisible to direct-request and capability metrics | HB greedy 0.5%, GSM 91.0, XSTest 6.8/10.0 | Supported |
| Update magnitude is necessary (dose threshold) | P2 scale pilot; anchor s=0.25 Base-level, s=1 35/41% | Supported (descriptive) |
| Teachers do not prefer harmful text at clean starts | P1 fixed-state logP; compliant prefix raises all | Supported (fixed-state only) |
| Plain-frozen repairs the vulnerability | R1 3.14% vs anchor 57.23%, paired CI; Base-level @16 | Supported |
| Stronger plain-prompt KL does not repair | R2 59.75%, R2-anchor @16 n.s. | Supported |
| Vulnerability generalizes to instruction-override attacks | idx8 0% for Base/anchor/R1 | Not supported (negative) |
| Plain-frozen generalizes across model families | Qwen2.5 attack ceiling; direct ASR worsens | Not established |
| Strong KL causally increases ASR | R2 vs anchor difference confounded | Not claimed |

## Reproduction commands (abridged)

```bash
PY=/data4/sjj/catelogue/alignment/safety-component/.venv-v31/bin/python
# R1 plain-frozen
$PY -m sdsd_safety.train --config configs/qwen3_4b.yaml \
  --run-id sdsd_v3_anchor_plainfrozen_100_seed42 \
  --override train.method=sdsd --override train.frozen_teacher_weight=0.25 \
  --override train.frozen_teacher_demo=false --override train.ref_kl_coef=0.03 \
  --override train.teacher_mode=ema --override train.num_demos=1 \
  --override train.demo_source=self --override train.lr=5e-5 \
  --override train.steps=100 --override seed=42
# Fixed attack (idx6 held-out)
$PY -m sdsd_safety.harmbench_attack --config configs/qwen3_4b.yaml --phase all \
  --jailbreak-index 6 --behaviors-csv <harmbench_behaviors_text_test.csv> \
  --expected-behaviors 159 --samples 16 \
  --protocol-id sdsd-safety-hb-humanjb-idx6-test159-v1 \
  --attack-dir results/raw_attack/humanjb-idx6-test159-v1 \
  --run <run_id>:<adapter_or_none>
```

Raw outputs, logs, weights and the environment directory are not committed; see
`.gitignore` and `status.md` for full commands, PIDs and provenance.
