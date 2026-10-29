"""Loss functions: token-level KL distillation, SFT, and GRPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_kl(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    direction: str = "reverse",
    mixed_weight: float = 0.5,
) -> torch.Tensor:
    """Average per-token KL over all completion tokens.

    direction: reverse = KL(student || teacher); forward = KL(teacher || student);
    mixed = mixed_weight * forward + (1 - mixed_weight) * reverse (OPSA default).
    """
    total = None
    count = 0
    for student, teacher in zip(student_logits, teacher_logits, strict=True):
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
        total = value.sum() if total is None else total + value.sum()
        count += value.shape[0]
    return total / max(count, 1)


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
    ref_logprobs: list[torch.Tensor],
    advantages: list[float],
    clip_eps: float = 0.2,
    ref_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    total: torch.Tensor | None = None
    count = 0
    clip_hits = 0.0
    for logits, ids, old, ref, advantage in zip(
        student_logits, token_ids, old_logprobs, ref_logprobs, advantages, strict=True
    ):
        new = F.log_softmax(logits.float(), dim=-1).gather(
            -1, torch.tensor(ids, dtype=torch.long, device=logits.device).unsqueeze(-1)
        ).squeeze(-1)
        ratio = torch.exp(new - old.detach())
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy = -torch.min(unclipped, clipped)
        ref_kl = new - ref.detach()
        value = policy + ref_coef * ref_kl
        total = value.sum() if total is None else total + value.sum()
        count += value.numel()
        clip_hits += float(((ratio - 1.0).abs() > clip_eps).float().mean())
    loss = total / max(count, 1)
    return loss, {"clip_fraction": clip_hits / max(len(student_logits), 1)}


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )
