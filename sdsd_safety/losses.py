"""Loss functions: token-level KL distillation, SFT, and GRPO."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def token_kl(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    direction: str = "reverse",
    mixed_weight: float = 0.5,
) -> torch.Tensor:
    """Mean of each completion's mean-token KL (padding is already excluded).

    direction: reverse = KL(student || teacher); forward = KL(teacher || student);
    mixed = mixed_weight * forward + (1 - mixed_weight) * reverse (OPSA default).
    """
    if not student_logits:
        raise ValueError("KL requires at least one nonempty completion")
    values = []
    for student, teacher in zip(student_logits, teacher_logits, strict=True):
        if student.shape != teacher.shape or student.ndim != 2 or student.shape[0] == 0:
            raise ValueError("student/teacher logits must have the same nonempty [tokens, vocab] shape")
        s = student.float()
        t = teacher.float().detach()
        log_s = F.log_softmax(s, dim=-1)
        log_t = F.log_softmax(t, dim=-1)
        reverse_kl = (log_s.exp() * (log_s - log_t)).sum(-1)
        forward_kl = (log_t.exp() * (log_t - log_s)).sum(-1)
        if direction == "reverse":
            value = reverse_kl
        elif direction == "forward":
            value = forward_kl
        elif direction == "mixed":
            value = mixed_weight * forward_kl + (1.0 - mixed_weight) * reverse_kl
        else:
            raise ValueError(f"unknown KL direction: {direction}")
        values.append(value.mean())
    return torch.stack(values).mean()


def mixture_teacher_log_probs(
    ema_logits: list[torch.Tensor],
    frozen_logits: list[torch.Tensor],
    frozen_weight: float,
) -> list[torch.Tensor]:
    """Detached log probabilities of (1-w)*EMA + w*frozen, not logit averaging.

    The trainer uses this for EMA/frozen mixing and, when selected, incremental
    probability pooling across demonstrations. It does not average KL losses.
    """
    weight = float(frozen_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("frozen_teacher_weight must lie in [0, 1]")
    output = []
    for ema, frozen in zip(ema_logits, frozen_logits, strict=True):
        if ema.shape != frozen.shape:
            raise ValueError("teacher mixture logits must have matching shapes")
        if weight == 0.0:
            output.append(F.log_softmax(ema.detach().float(), dim=-1))
        elif weight == 1.0:
            output.append(F.log_softmax(frozen.detach().float(), dim=-1))
        else:
            log_ema = F.log_softmax(ema.detach().float(), dim=-1)
            log_frozen = F.log_softmax(frozen.detach().float(), dim=-1)
            output.append(torch.logaddexp(log_ema + math.log1p(-weight), log_frozen + math.log(weight)))
    return output


def gathered_logprobs(logits: list[torch.Tensor], token_ids: list[list[int]]) -> list[torch.Tensor]:
    values = []
    for row, ids in zip(logits, token_ids, strict=True):
        index = torch.tensor(ids, dtype=torch.long, device=row.device)
        values.append(F.log_softmax(row.float(), dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1))
    return values


def grpo_loss(
    student_logits: list[torch.Tensor],
    token_ids: list[list[int]],
    old_logprobs: list[torch.Tensor],
    reference_logits: list[torch.Tensor],
    advantages: list[float],
    clip_eps: float = 0.2,
    ref_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not student_logits:
        raise ValueError("GRPO requires at least one completion")
    policies = []
    clip_hits = 0.0
    for logits, ids, old, advantage in zip(
        student_logits, token_ids, old_logprobs, advantages, strict=True
    ):
        new = F.log_softmax(logits.float(), dim=-1).gather(
            -1, torch.tensor(ids, dtype=torch.long, device=logits.device).unsqueeze(-1)
        ).squeeze(-1)
        ratio = torch.exp(new - old.detach())
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy = -torch.min(unclipped, clipped)
        policies.append(policy.mean())
        clip_hits += float(((ratio - 1.0).abs() > clip_eps).float().mean())
    reference_kl = token_kl(student_logits, reference_logits, direction="reverse")
    loss = torch.stack(policies).mean() + ref_coef * reference_kl
    return loss, {
        "clip_fraction": clip_hits / len(student_logits),
        "reference_kl": float(reference_kl.detach()),
    }


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )
