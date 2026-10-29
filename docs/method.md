# SDSD-Safety: method and safety adaptation

## 1. Background

SDSD (**S**elf-**D**istillation with **S**ampled **D**emonstrations;
Nicolicioiu, Pezeshki & Courville, arXiv:2606.26091) studies the on-policy
self-distillation objective used by SDPO and related methods. For a question
`x`, the student samples `N` rollouts `y ~ pi_theta(.|x)`. A demonstration
`y_corr` is sampled from the set of verified-correct rollouts and given to an
EMA teacher. The teacher scores the student's own prefix:

```
L_SD(theta; x) = E_{y_corr} [ (1/N) sum_y sum_t
                  KL( pi_theta(.|x, y_<t) || sg[ pi_theta_bar(.|x, y_corr, y_<t)] ) ]
```

The paper's central result is that the optimum of SDSD with KL regularization
tilts the base distribution by the expected pointwise conditional mutual
information (PCMI)

```
pi*_SD-KL(y|x) ∝ pi_0(y|x) exp( (1/(1+beta)) E_{y_corr} log[ pi_0(y|x,y_corr) / pi_0(y|x) ] )
```

rather than by a task reward. Consequently, among equally correct outputs, a
rollout that is more compatible with the sampled demonstration is reinforced
more strongly, which can concentrate probability mass on already-dominant
modes. The paper's experiments are on graph path finding and science QA; its
diversity analysis is not a safety claim.

## 2. Safety adaptation

The safety task replaces "verified-correct rollout" with "qualified safe
rollout". The adaptation keeps the original objective and changes only the
qualification rule and the role of the demonstration.

### 2.1 Qualification

A rollout is qualified when Qwen3Guard-Gen-4B returns `Safety: Safe` for the
(prompt, response) pair.

- Harmful prompt: a qualified rollout is a refusal or a safe alternative
  response. Any safe behavior is accepted.
- Normal prompt: a qualified rollout must additionally be non-refusing
  (official XSTest refusal prefixes) and at least 4 tokens long. This prevents
  degenerate refusals from becoming "good" normal-task demonstrations.

No new scorer, controller, gate, reward model, or training stage is introduced.
The judge is a local, frozen, off-the-shelf guard model already present on the
machine.

### 2.2 Demonstration sampling and fallback

For each prompt, the student generates `N=4` rollouts at temperature 1.0. The
qualified rollouts are shuffled and up to `K` distinct texts are used as teacher
demonstrations. If fewer than `K` are available, the external safe/helpful
response (PKU safe response for harmful prompts, alpaca response for normal
prompts) fills the remaining slots. If none is available, the teacher is
conditioned on the question without a demonstration; this is logged through the
`demo_self_rate` metric.

This fallback is deliberately simple and fixed before the main runs. It is the
same external response used by the SFT baseline.

### 2.3 Teacher context

The teacher sees the question, the demonstration as a prior assistant turn, and
a short instruction to answer the original request again. The student's own
rollout tokens are appended after the generation prompt, and the teacher's
next-token distributions on exactly those tokens are used:

```
<user> question
<assistant> demonstration
<user> Now respond to my original request.
<assistant> [student prefix]
```

Three demonstrations (`K=3`) are not concatenated. Each demonstration
constructs an independent teacher context, and the per-token distillation
losses are averaged, matching the paper's K=3 ablation.

### 2.4 Teacher and reference

The teacher is an EMA of the student's LoRA parameters with update rate 0.05
(`theta_bar <- 0.95 theta_bar + 0.05 theta`) after each optimizer step. Because
training is LoRA-only, the EMA is over the trainable adapter parameters; the
frozen base weights are shared conceptually but kept as a second model copy for
implementation simplicity. The reference model is the frozen base model. A
token-level reverse KL to the reference is added with coefficient 0.01 for SDSD
and GRPO.

### 2.5 Objective

SDSD-Safety minimizes the token-level reverse KL from the student to the
demonstration-conditioned EMA teacher, averaged over rollouts and
demonstrations, plus the reference KL:

```
L = mean_{y_corr, y} mean_t KL( pi_theta(.|prefix) || sg pi_theta_bar(.|x, y_corr, prefix) )
    + beta_ref mean_t KL( pi_theta(.|prefix) || sg pi_ref(.|prefix) )
```

### 2.6 OPSA baseline

OPSA (arXiv:2605.15239) uses a frozen copy of the base model as teacher and a
type-conditional privileged system prompt:

- harmful: "This prompt contains a harmful request. Your only allowed action is
  to refuse. You must NOT: ... Respond with a brief, clear refusal only."
- benign: "This is a safe, benign request. Respond helpfully and directly. You
  must NOT refuse, add safety disclaimers, or treat this as potentially
  harmful. Provide a complete, accurate answer."

The student is rolled out without this system prompt. OPSA's default loss is
`mixed` KL, 0.5 forward + 0.5 reverse; we use the full vocabulary rather than a
top-k truncation. The teacher is frozen, not EMA, and there is no explicit
reference KL, matching OPSA's design.

### 2.7 GRPO baseline

Standard group-relative policy optimization with 4 rollouts per prompt and a
refusal-aware Qwen3Guard reward: harmful prompt `+1` if Safe and refusing,
`0` if Safe but non-refusing, `-1` if unsafe; normal prompt `+1` if Safe and
non-refusing else `-1`. The refusal term is needed because the base model is
already Safe on ~98% of PKU rollouts, so a purely binary Safe/Unsafe reward
gives zero within-group variance on almost every prompt (only 1/100 pilot
groups had mixed labels). Advantages are normalized within each prompt group;
the token-level clipped objective uses epsilon 0.2 and a reference KL
coefficient of 0.01.

### 2.8 SFT baseline

Assistant-token cross-entropy on the same external responses used as SDSD
fallback demonstrations (PKU safe response for harmful prompts, alpaca response
for normal prompts). No judge filtering is applied to these fixed targets.

## 3. Evaluation protocol

- **HarmBench**: all 400 text behaviors, one greedy generation per behavior,
  official Llama-2-13B behavior classifier, ASR = fraction labeled "yes".
- **XSTest**: all 450 official prompts, greedy generation, official refusal
  string matching. Report refusal rate on the 200 unsafe prompts and refusal
  rate on the 250 safe prompts (over-refusal).
- **GSM8K**: 200 fixed test questions (seed 42), greedy generation, exact match
  of the number after `####`.
- **Dev set**: 300 held-out PKU harmful prompts + 100 held-out alpaca prompts;
  used only for checkpoint/run selection, never for training. Test benchmarks
  are never used for training or selection.

## 4. Theory mapping for the safety result

The safety setting is a special case of SDSD's analysis: among safe responses,
the EMA teacher conditioned on a safe demonstration will assign higher
probability to responses that share the demonstration's refusal style or safe
completion strategy. The SDSD tilt therefore predicts that safety training
concentrates mass on the dominant safe mode. Two consequences are checked in
the experiments:

1. HarmBench ASR should fall relative to Base and SFT if the safe mode is
   reinforced.
2. XSTest over-refusal and GSM8K accuracy test whether the reinforced mode is
   an over-generalized refusal mode or a task-conditioned safety behavior.

The theory does not predict that SDSD is uniformly better than GRPO; the
comparison is empirical. The project reports the measured trade-off rather than
assuming a win.

## 5. Deviations and limitations

- 4B non-thinking backbone and 250 steps, not the paper's 8B/30-epoch QA scale.
- LoRA, not full-parameter fine-tuning; EMA is over adapter parameters.
- One seed for the main table, with at most two additional exploratory seeds
  recorded separately if needed.
- Harmless/helpful behavior is represented by alpaca-cleaned; this is a broad
  but imperfect normal-task proxy.
- The Qwen3Guard judge is also the GRPO reward source; the evaluation of safety
  uses an independent official HarmBench classifier and rule-based XSTest, but
  the judge's blind spots can enter training.
