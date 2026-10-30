"""Unified trainer for SFT, GRPO, OPSA, SIRL and SDSD-Safety.

All methods share the model, data split, LoRA parameterization, decoding
settings and evaluation code. Only the objective differs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .judging import Qwen3GuardJudge, judge_summary
from .losses import (
    full_vocab_entropy_rewards, gathered_logprobs, grpo_loss,
    mixture_teacher_log_probs, sft_loss, token_kl,
)
from .metrics import is_refusal
from .modeling import (
    attach_lora,
    chat_prompt,
    completion_logits,
    copy_lora,
    ema_lora,
    generate_rollouts,
    load_base_model,
    load_tokenizer,
    teacher_prompt,
)
from .utils import (
    append_jsonl,
    gpu_memory_gib,
    load_config,
    now_cst,
    read_jsonl,
    resolve_path,
    set_seed,
    sha256_file,
    setup_logger,
    write_json,
)

OPSA_HARMFUL_PROMPT = (
    "This prompt contains a harmful request. Your only allowed action is to refuse. "
    "You must NOT: provide the requested content in any form; offer a safer or educational version; "
    "discuss the topic from an academic perspective; partially comply while adding disclaimers. "
    "Respond with a brief, clear refusal only."
)
OPSA_BENIGN_PROMPT = (
    "This is a safe, benign request. Respond helpfully and directly. "
    "You must NOT refuse, add safety disclaimers, or treat this as potentially harmful. "
    "Provide a complete, accurate answer."
)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_worktree_status() -> dict[str, Any]:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"git_dirty": bool(status), "git_status": status}
    except Exception:
        return {"git_dirty": None, "git_status": None}


def stratified_steps(rows: list[dict[str, Any]], batch: int, steps: int, seed: int, harmful_fraction: float | None = None):
    """Yield deterministic batches, with an explicit optional risk/benign mix."""
    harmful = [row for row in rows if row["kind"] == "harmful"]
    normal = [row for row in rows if row["kind"] == "normal"]
    if not rows or batch < 1 or steps < 1:
        raise ValueError("training rows, batch size and steps must be nonempty/positive")
    fraction = len(harmful) / len(rows) if harmful_fraction is None else float(harmful_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("harmful_batch_fraction must lie in [0, 1]")
    n_harmful = round(batch * fraction)
    n_normal = batch - n_harmful
    if (n_harmful and not harmful) or (n_normal and not normal):
        raise ValueError("requested batch mix requires both populated training pools")
    rng = random.Random(seed)
    pools = {name: items[:] for name, items in (("harmful", harmful), ("normal", normal))}
    for items in pools.values():
        rng.shuffle(items)
    cursors = {"harmful": 0, "normal": 0}
    for _ in range(steps):
        batch_rows = []
        for name, count in (("harmful", n_harmful), ("normal", n_normal)):
            pool = pools[name]
            for _ in range(count):
                if cursors[name] >= len(pool):
                    rng.shuffle(pool)
                    cursors[name] = 0
                batch_rows.append(pool[cursors[name]])
                cursors[name] += 1
        rng.shuffle(batch_rows)
        yield batch_rows


def response_is_refusal(text: str, judgment: dict[str, Any]) -> bool:
    """Use a semantic refusal label when available; otherwise use the rule."""
    label = judgment.get("refusal")
    return label if isinstance(label, bool) else is_refusal(text)


def verified_refusal_target(row: dict[str, Any]) -> str:
    target = row.get("sft_response")
    if not target or row.get("sft_safety") != "Safe" or row.get("sft_refusal") is not True:
        raise ValueError(f"row {row.get('id')} lacks a verified safe refusal target; run make_sft_targets")
    return target


def verified_normal_target(row: dict[str, Any]) -> str:
    target = row.get("external_response")
    if not target or row.get("external_safety") != "Safe" or row.get("external_refusal") is not False:
        raise ValueError(f"row {row.get('id')} lacks an audited non-refusal external answer")
    return target


def select_demos(
    row: dict[str, Any],
    samples: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[str], int]:
    """Return (demonstrations, number taken from the student's own rollouts)."""
    cfg = config["train"]
    min_tokens = int(config["data"].get("min_demo_tokens", 4))
    num_demos = int(cfg.get("num_demos", 1))
    if num_demos < 1:
        raise ValueError("num_demos must be positive")
    strict_refusal = row["kind"] == "harmful" and cfg.get("demo_preference", "refusal") == "refusal"
    source = cfg.get("demo_source", "self")
    if source not in {"self", "external", "external_refusal"}:
        raise ValueError(f"unknown demo_source {source}")
    if row["kind"] == "normal":
        external = verified_normal_target(row)
        source = cfg.get("normal_demo_source", "external")
        if source not in {"self", "external"}:
            raise ValueError("normal_demo_source must be self or external")
    elif strict_refusal or source == "external_refusal":
        external = verified_refusal_target(row)
    else:
        external = row["external_response"]
        if row.get("external_safety") != "Safe":
            raise ValueError("external demonstration has not passed the safety audit")
    if source in {"external", "external_refusal"}:
        return [external] * num_demos, 0
    qualified = []
    seen = set()
    for sample, judgment in zip(samples, judgments, strict=True):
        text = sample["text"].strip()
        safe = judgment.get("safety") == "Safe" and bool(text)
        refusal = response_is_refusal(text, judgment)
        if row["kind"] == "normal":
            safe = safe and not refusal and len(sample["token_ids"]) >= min_tokens
        elif strict_refusal:
            safe = safe and refusal
        if safe and text not in seen:
            qualified.append(text)
            seen.add(text)
    rng.shuffle(qualified)
    demos = qualified[:num_demos]
    used_self = len(demos)
    if len(demos) < num_demos:
        demos += [external] * (num_demos - len(demos))
    return demos, used_self


@torch.no_grad()
def evaluate_dev(model, tokenizer, judge: Qwen3GuardJudge, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    prompts = [row["prompt"] for row in rows]
    rollouts = generate_rollouts(
        model,
        tokenizer,
        prompts,
        num_rollouts=1,
        max_new_tokens=int(config["data"]["max_new_tokens"]),
        batch_size=min(8, int(config["eval"]["batch_size"])),
        greedy=True,
        max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
    )
    responses = [group[0]["text"] for group in rollouts]
    judgments = judge.judge_responses(prompts, responses)
    harmful = [j for row, j in zip(rows, judgments, strict=True) if row["kind"] == "harmful"]
    normal = [j for row, j in zip(rows, judgments, strict=True) if row["kind"] == "normal"]
    normal_refusal = [
        response_is_refusal(response, judgment)
        for row, response, judgment in zip(rows, responses, judgments, strict=True)
        if row["kind"] == "normal"
    ]
    return {
        "harmful": judge_summary(harmful),
        "normal": judge_summary(normal),
        "normal_refusal_rate": sum(normal_refusal) / max(len(normal_refusal), 1),
        "n": len(rows),
    }


def update_run(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(updates)
    payload["updated_at"] = now_cst()
    write_json(path, payload)
    return payload


def load_training_data(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Require the audited training artifact, without opening evaluation data."""
    data = config["data"]
    train_key = "sft_train" if config["train"]["method"] == "sft" and data.get("sft_train") else "train"
    path = Path(resolve_path(config, data[train_key])).resolve()
    for split in ("dev", "test"):
        if data.get(split) and path == Path(resolve_path(config, data[split])).resolve():
            raise ValueError(f"training path must not equal the {split} path")
    if not data.get("manifest"):
        raise ValueError("data.manifest is required for audited training")
    manifest_path = Path(resolve_path(config, data["manifest"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("prompt_audit", {})
    if audit.get("completed") is not True or audit.get("protocol") != "prompt-intent-v2":
        raise ValueError("training requires a completed prompt-intent-v2 audit")
    if manifest.get("targets", {}).get("completed") is not True:
        raise ValueError("training requires verified refusal targets; run make_sft_targets")
    file_info = manifest.get("files", {}).get(train_key, {})
    if not file_info.get("path") or Path(resolve_path(config, file_info["path"])).resolve() != path:
        raise ValueError(f"manifest does not identify the selected {train_key} artifact")
    digest = sha256_file(path)
    if digest != file_info.get("sha256"):
        raise ValueError("training data hash does not match the audited manifest")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("training artifact is empty")
    for row in rows:
        if row.get("kind") == "harmful":
            verified_refusal_target(row)
        elif row.get("kind") == "normal":
            verified_normal_target(row)
        else:
            raise ValueError(f"row {row.get('id')} has an unaudited training kind")
    return rows, {
        "data_path": str(path), "data_manifest": str(manifest_path),
        "data_sha256": digest, "data_manifest_sha256": sha256_file(manifest_path),
    }


def set_step_learning_rate(optimizer, config: dict[str, Any], step: int) -> float:
    warmup = int(config.get("warmup_steps", 0))
    if warmup < 0:
        raise ValueError("warmup_steps must be nonnegative")
    scale = min(step / warmup, 1.0) if warmup else 1.0
    lr = float(config["lr"]) * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_training_checkpoint(student, teacher, run_dir: Path, step: int, *, completed: bool = False) -> None:
    """Overwrite one student and, when used, one EMA adapter; never save base weights."""
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if completed and (metadata.get("data_verified") is not True or step != metadata.get("steps_requested")):
        raise ValueError("a final valid checkpoint requires verified data and all requested steps")
    checkpoint = run_dir / "checkpoint"
    student.save_pretrained(checkpoint)
    if teacher is not None:
        teacher.save_pretrained(run_dir / "teacher_checkpoint")
    write_json(checkpoint / "training_step.json", {"step": step, "training_completed": completed})
    updates = {
        "checkpoint": str(checkpoint), "last_checkpoint_step": step, "steps_completed": step,
        "checkpoint_kind": "adapter_only_no_optimizer_resume",
    }
    if completed:
        updates.update(status="trained", training_completed=True, valid=True, trained_at=now_cst())
    update_run(run_dir, **updates)


def record_training_step(record, student, teacher, config, run_dir) -> None:
    append_jsonl(run_dir / "metrics.jsonl", record)
    update_run(run_dir, steps_completed=record["step"], final_loss=record["loss"])
    interval = int(config["train"].get("save_every", 50))
    if interval < 0:
        raise ValueError("save_every must be nonnegative")
    if interval and record["step"] % interval == 0 and record["step"] < int(config["train"]["steps"]):
        save_training_checkpoint(student, teacher, run_dir, record["step"])


def train_sft(student, tokenizer, judge, rows, config, run_dir, logger) -> list[dict[str, Any]]:
    from .modeling import encode_sft_pair

    cfg = config["train"]
    batch = int(cfg["batch_prompts"])
    micro = min(4, batch)
    max_length = int(config["data"]["max_prompt_tokens"]) + int(config["data"]["max_new_tokens"])
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    history = []
    batches = stratified_steps(rows, batch, int(cfg["steps"]), int(config["seed"]), cfg.get("harmful_batch_fraction"))
    for step, batch_rows in enumerate(batches, start=1):
        started = time.time()
        learning_rate = set_step_learning_rate(optimizer, cfg, step)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        encoded = [
            encode_sft_pair(
                tokenizer, row["prompt"],
                verified_refusal_target(row) if row["kind"] == "harmful" else verified_normal_target(row),
                max_length,
            )
            for row in batch_rows
        ]
        if any(item["prompt_tokens"] > int(config["data"]["max_prompt_tokens"]) for item in encoded):
            raise ValueError("an SFT question exceeds the explicit prompt budget")
        target_tokens = sum(sum(value != -100 for value in item["labels"][1:]) for item in encoded)
        if target_tokens == 0:
            raise ValueError("SFT batch has no supervised response tokens")
        loss_value = 0.0
        for start in range(0, len(encoded), micro):
            chunk = encoded[start : start + micro]
            width = max(len(item["input_ids"]) for item in chunk)
            input_ids = torch.full(
                (len(chunk), width), tokenizer.pad_token_id, dtype=torch.long, device=model_device(student)
            )
            labels = torch.full((len(chunk), width), -100, dtype=torch.long, device=model_device(student))
            attention = torch.zeros((len(chunk), width), dtype=torch.long, device=model_device(student))
            for row_index, item in enumerate(chunk):
                length = len(item["input_ids"])
                input_ids[row_index, :length] = torch.tensor(item["input_ids"], device=input_ids.device)
                labels[row_index, :length] = torch.tensor(item["labels"], device=input_ids.device)
                attention[row_index, :length] = 1
            logits = student(input_ids=input_ids, attention_mask=attention).logits
            loss = sft_loss(logits[:, :-1], labels[:, 1:])
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("nonfinite SFT loss")
            weight = int((labels[:, 1:] != -100).sum().item()) / target_tokens
            (loss * weight).backward()
            loss_value += float(loss.item()) * weight
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], float(cfg["grad_clip"])
        )
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError("nonfinite SFT gradient norm")
        optimizer.step()
        record = {
            "step": step,
            "loss": loss_value,
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm),
            "seconds": time.time() - started,
        }
        history.append(record)
        record_training_step(record, student, None, config, run_dir)
        if step % int(cfg["log_every"]) == 0:
            logger.info(
                "step %d/%d loss=%.4f grad=%.3f sec=%.1f",
                step,
                int(cfg["steps"]),
                record["loss"],
                record["grad_norm"],
                record["seconds"],
            )
    return history


def train_onpolicy(student, teacher, tokenizer, judge, rows, config, run_dir, logger) -> list[dict[str, Any]]:
    cfg = config["train"]
    method = cfg["method"]
    n = int(cfg["num_rollouts"])
    batch = int(cfg["batch_prompts"])
    micro = min(4 if method in {"sdsd", "opsa"} else 8, batch * n)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    rng = random.Random(int(config["seed"]) + 17)
    frozen_weight = float(cfg.get("frozen_teacher_weight", 0.0))
    pooling = cfg.get("demo_pooling", "mean_kl")
    frozen_teacher_demo = bool(cfg.get("frozen_teacher_demo", True))
    if not 0.0 <= frozen_weight <= 1.0 or pooling not in {"mean_kl", "probability_mean"}:
        raise ValueError("invalid teacher mixture or demonstration pooling configuration")
    if method != "sdsd" and not frozen_teacher_demo:
        raise ValueError("frozen_teacher_demo=False is only defined for SDSD")
    if method == "sdsd" and teacher is None and frozen_weight != 1.0:
        raise ValueError("an EMA component was requested but no EMA teacher was initialized")
    history = []
    batches = stratified_steps(rows, batch, int(cfg["steps"]), int(config["seed"]), cfg.get("harmful_batch_fraction"))
    for step, batch_rows in enumerate(batches, start=1):
        started = time.time()
        learning_rate = set_step_learning_rate(optimizer, cfg, step)
        student.eval()
        prompts = [row["prompt"] for row in batch_rows]
        rollouts = generate_rollouts(
            student,
            tokenizer,
            prompts,
            num_rollouts=n,
            max_new_tokens=int(config["data"]["max_new_tokens"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            batch_size=min(8, batch),
            max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
        )
        if any(len(samples) != n for samples in rollouts):
            raise RuntimeError("generation returned a different number of rollouts than requested")
        flat_rows = [row for row, samples in zip(batch_rows, rollouts, strict=True) for _ in samples]
        flat_samples = [sample for samples in rollouts for sample in samples]
        flat_judgments = judge.judge_responses(
            [row["prompt"] for row in flat_rows], [sample["text"] for sample in flat_samples],
        )
        if len(flat_judgments) != len(flat_samples):
            raise RuntimeError("judge returned a different number of labels than rollouts")
        flat_demos: list[list[str]] = []
        flat_demo_self: list[int] = []
        for row_index, (row, samples) in enumerate(zip(batch_rows, rollouts, strict=True)):
            judgments = flat_judgments[row_index * n : (row_index + 1) * n]
            if method == "sdsd":
                for _ in range(n):
                    demos, used_self = select_demos(row, samples, judgments, config, rng)
                    flat_demos.append(demos)
                    flat_demo_self.append(used_self)

        safe_flags = [judgment.get("safety") == "Safe" for judgment in flat_judgments]
        refusal_flags = [
            response_is_refusal(sample["text"], judgment)
            for sample, judgment in zip(flat_samples, flat_judgments, strict=True)
        ]
        completion_ids = [sample["token_ids"] for sample in flat_samples]
        student_prompts = [chat_prompt(tokenizer, row["prompt"]) for row in flat_rows]

        # eval() disables policy dropout but does not disable autograd.
        student.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        extra: dict[str, float] = {}
        demo_self = 0
        demo_total = 0
        advantages: list[float] = []
        old_logprobs: list[torch.Tensor] = []
        entropy_reward_mean = None

        if method in {"grpo", "sirl"}:
            student.eval()
            rewards: list[float] = []
            entropy_rewards: list[float] = []
            with torch.no_grad():
                for start in range(0, len(flat_rows), micro):
                    ids = completion_ids[start : start + micro]
                    old_logits = completion_logits(
                        student, tokenizer, student_prompts[start : start + micro], ids,
                        with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                    )
                    old_logprobs.extend(gathered_logprobs(old_logits, ids))
                    if method == "sirl":
                        entropy_rewards.extend(full_vocab_entropy_rewards(old_logits))
                    del old_logits
            if method == "grpo":
                margins = judge.score_responses(
                    [row["prompt"] for row in flat_rows],
                    [sample["text"] for sample in flat_samples],
                )
                for row, margin, refusal in zip(flat_rows, margins, refusal_flags, strict=True):
                    if row["kind"] == "harmful":
                        rewards.append(float(margin) + (float(cfg.get("grpo_harmful_refusal_bonus", 4.0)) if refusal else 0.0))
                    else:
                        rewards.append(float(margin) - (float(cfg.get("grpo_normal_refusal_penalty", 8.0)) if refusal else 0.0))
            else:
                if len(entropy_rewards) != len(flat_rows):
                    raise RuntimeError("SIRL entropy reward count differs from rollouts")
                # No guard margin, refusal bonus, or label term is added.
                rewards = entropy_rewards
                entropy_reward_mean = sum(entropy_rewards) / len(entropy_rewards)
            for start in range(0, len(flat_rows), n):
                group = rewards[start : start + n]
                mean = sum(group) / len(group)
                variance = sum((value - mean) ** 2 for value in group) / len(group)
                std = math.sqrt(variance + 1e-6)
                advantages.extend((value - mean) / std for value in group)

        for start in range(0, len(flat_rows), micro):
            chunk_rows = flat_rows[start : start + micro]
            chunk_ids = completion_ids[start : start + micro]
            chunk_prompts = student_prompts[start : start + micro]
            chunk_weight = len(chunk_rows) / len(flat_rows)
            logits = completion_logits(
                student, tokenizer, chunk_prompts, chunk_ids,
                micro_batch=min(4, len(chunk_rows)), with_grad=True,
                max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
            )
            if method in {"grpo", "sirl"}:
                with torch.no_grad(), student.disable_adapter():
                    ref_logits = completion_logits(
                        student, tokenizer, chunk_prompts, chunk_ids,
                        with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                    )
                loss, stats = grpo_loss(
                    logits,
                    chunk_ids,
                    old_logprobs[start : start + micro],
                    ref_logits,
                    advantages[start : start + micro],
                    ref_coef=float(cfg["ref_kl_coef"]),
                )
                for name, value in stats.items():
                    extra[name] = extra.get(name, 0.0) + value * chunk_weight
            elif method == "opsa":
                teacher_prompts = [
                    chat_prompt(
                        tokenizer,
                        row["prompt"],
                        system=OPSA_HARMFUL_PROMPT if row["kind"] == "harmful" else OPSA_BENIGN_PROMPT,
                    )
                    for row in chunk_rows
                ]
                student.eval()
                with torch.no_grad(), student.disable_adapter():
                    teacher_logits = completion_logits(
                        student, tokenizer, teacher_prompts, chunk_ids,
                        with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                    )
                loss = token_kl(
                    logits, teacher_logits, direction="mixed",
                    mixed_weight=float(cfg["opsa_mixed_kl_weight"]),
                )
            elif method == "sdsd":
                demo_lists = flat_demos[start : start + micro]
                demo_self += sum(flat_demo_self[start : start + micro])
                demo_total += sum(len(demos) for demos in demo_lists)
                per_demo = []
                pooled_logits = None
                for demo_index in range(len(demo_lists[0])):
                    teacher_prompts = [
                        teacher_prompt(
                            tokenizer, row["prompt"], demos[demo_index],
                            max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                        )
                        for row, demos in zip(chunk_rows, demo_lists, strict=True)
                    ]
                    with torch.no_grad():
                        if teacher is not None:
                            ema_logits = completion_logits(
                                teacher, tokenizer, teacher_prompts, chunk_ids,
                                with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                            )
                        if teacher is None or frozen_weight > 0.0:
                            frozen_contexts = teacher_prompts if frozen_teacher_demo else [
                                chat_prompt(tokenizer, row["prompt"]) for row in chunk_rows
                            ]
                            with student.disable_adapter():
                                frozen_logits = completion_logits(
                                    student, tokenizer, frozen_contexts, chunk_ids,
                                    with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                                )
                        if teacher is None:
                            teacher_logits = frozen_logits
                        elif frozen_weight == 0.0:
                            teacher_logits = ema_logits
                        else:
                            teacher_logits = mixture_teacher_log_probs(ema_logits, frozen_logits, frozen_weight)
                    if pooling == "mean_kl":
                        per_demo.append(token_kl(logits, teacher_logits, direction="reverse"))
                    else:
                        # Running arithmetic probability mean; K contexts stay independent.
                        pooled_logits = teacher_logits if pooled_logits is None else mixture_teacher_log_probs(
                            pooled_logits, teacher_logits, 1.0 / (demo_index + 1),
                        )
                loss = torch.stack(per_demo).mean() if pooling == "mean_kl" else token_kl(
                    logits, pooled_logits, direction="reverse",
                )
                student.eval()
                with torch.no_grad(), student.disable_adapter():
                    ref_logits = completion_logits(
                        student, tokenizer, chunk_prompts, chunk_ids,
                        with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                    )
                loss = loss + float(cfg["ref_kl_coef"]) * token_kl(logits, ref_logits, direction="reverse")
            else:
                raise ValueError(f"unknown method {method}")
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("nonfinite on-policy loss")
            (loss * chunk_weight).backward()
            loss_sum += float(loss.item()) * chunk_weight
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], float(cfg["grad_clip"])
        )
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError("nonfinite on-policy gradient norm")
        optimizer.step()
        if method == "sdsd" and teacher is not None:
            ema_lora(student, teacher, decay=1.0 - float(cfg["teacher_update_rate"]))

        harmful_idx = [i for i, row in enumerate(flat_rows) if row["kind"] == "harmful"]
        normal_idx = [i for i, row in enumerate(flat_rows) if row["kind"] == "normal"]
        record = {
            "step": step,
            "loss": loss_sum,
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm),
            "rollout_safe_rate": sum(safe_flags) / len(safe_flags),
            "rollout_refusal_rate": sum(refusal_flags) / len(refusal_flags),
            "harmful_refusal_rate": sum(refusal_flags[i] for i in harmful_idx) / max(len(harmful_idx), 1),
            "normal_refusal_rate": sum(refusal_flags[i] for i in normal_idx) / max(len(normal_idx), 1),
            "mean_completion_tokens": sum(len(ids) for ids in completion_ids) / len(completion_ids),
            "demo_self_rate": demo_self / max(demo_total, 1) if method == "sdsd" else 0.0,
            "seconds": time.time() - started,
        }
        record.update(extra)
        if entropy_reward_mean is not None:
            record["entropy_reward_mean"] = entropy_reward_mean
            record["guard_margin_used"] = False
        history.append(record)
        record_training_step(record, student, teacher, config, run_dir)
        if step % int(cfg["log_every"]) == 0:
            logger.info(
                "step %d/%d loss=%.4f grad=%.3f safe=%.2f refusal=%.2f len=%.0f self=%.2f sec=%.1f",
                step,
                int(cfg["steps"]),
                record["loss"],
                record["grad_norm"],
                record["rollout_safe_rate"],
                record["rollout_refusal_rate"],
                record["mean_completion_tokens"],
                record["demo_self_rate"],
                record["seconds"],
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    cfg = config["train"]
    method = cfg["method"]
    if method not in {"sft", "grpo", "opsa", "sirl", "sdsd"}:
        raise SystemExit(f"unknown method {method}")
    cfg.setdefault("demo_preference", "refusal")
    if cfg["demo_preference"] not in {"refusal", "any_safe"}:
        raise SystemExit("demo_preference must be refusal or any_safe")
    cfg.setdefault("normal_demo_source", "external")
    cfg.setdefault("demo_pooling", "mean_kl")
    cfg.setdefault("frozen_teacher_weight", 0.0)
    cfg.setdefault("frozen_teacher_demo", True)
    cfg.setdefault("save_every", 50)
    cfg.setdefault("teacher_mode", "ema")
    teacher_mode = cfg.get("teacher_mode", "ema")
    if teacher_mode not in {"ema", "frozen"}:
        raise SystemExit("teacher_mode must be ema or frozen")
    if teacher_mode == "frozen":
        cfg["frozen_teacher_weight"] = 1.0
    if not 0.0 <= float(cfg["frozen_teacher_weight"]) <= 1.0:
        raise SystemExit("frozen_teacher_weight must lie in [0, 1]")
    if cfg["demo_pooling"] not in {"mean_kl", "probability_mean"}:
        raise SystemExit("demo_pooling must be mean_kl or probability_mean")
    for key in ("train", "sft_train", "dev", "test", "manifest"):
        if config["data"].get(key):
            config["data"][key] = str(Path(resolve_path(config, config["data"][key])).resolve())
    if args.run_id:
        run_id = args.run_id
    elif method == "sdsd":
        run_id = f"sdsd_{cfg.get('demo_source', 'self')}_k{cfg.get('num_demos', 1)}_seed{config['seed']}"
    else:
        run_id = f"{method}_seed{config['seed']}"
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise SystemExit("run-id must be a single directory name")
    run_dir = Path(resolve_path(config, cfg["output_root"])) / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory exists: {run_dir}; use a fresh run-id (adapter-only checkpoints are not resumable)")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "train.log", name=f"sdsd.{run_id}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    update_run(
        run_dir, protocol_id="sdsd-safety-training-v3", status="running",
        training_completed=False, valid=False, run_id=run_id, method=method,
        seed=int(config["seed"]), steps_requested=int(cfg["steps"]), steps_completed=0,
        checkpoint=None, config=config, git_commit=git_commit(),
        command=shlex.join([sys.executable, *sys.argv]), argv=[sys.executable, *sys.argv],
        created_at=now_cst(), data_manifest_sha256=None, data_sha256=None,
        code_sha256={path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        source_config=str(Path(args.config).resolve()),
        loss_reduction="mean_supervised_token" if method == "sft" else "mean_completion_mean_token",
        policy_dropout="disabled_during_onpolicy_scoring",
        teacher_context=("reference_system_original_user_v2" if cfg.get("frozen_teacher_demo", True)
                         else "ema_demo_plus_frozen_plain_user"),
        entropy_reward="negative_mean_full_vocabulary_entropy_detached" if method == "sirl" else None,
        guard_margin_used=(method == "grpo"),
        optimizer_resume_supported=False,
        **git_worktree_status(),
    )
    logger.info("run_id=%s method=%s config=%s", run_id, method, json.dumps(cfg, sort_keys=True))
    logger.info("git_commit=%s", git_commit())

    def terminate(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        rows, provenance = load_training_data(config)
        update_run(run_dir, **provenance, data_verified=True, training_rows=len(rows))
        logger.info("audited training rows: %d data_sha256=%s", len(rows), provenance["data_sha256"])
        free, total = gpu_memory_gib()
        logger.info("gpu free=%.1f GiB total=%.1f GiB", free, total)
        if free < 25:
            raise RuntimeError(f"not enough free GPU memory: {free:.1f} GiB")
        set_seed(int(config["seed"]))
        tokenizer = load_tokenizer(config["model"]["path"])
        base = load_base_model(
            config["model"]["path"],
            dtype=config["model"]["dtype"],
            attn_implementation=config["model"]["attn_implementation"],
        )
        student = attach_lora(base, config["model"]["lora"])
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        update_run(run_dir, trainable_parameters=int(trainable))
        logger.info("trainable parameters: %d (%.2fM)", trainable, trainable / 1e6)

        teacher = None
        if method == "sdsd" and float(cfg["frozen_teacher_weight"]) < 1.0:
            teacher_base = load_base_model(
                config["model"]["path"], dtype=config["model"]["dtype"],
                attn_implementation=config["model"]["attn_implementation"],
            )
            teacher = attach_lora(teacher_base, config["model"]["lora"])
            copy_lora(student, teacher)
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            teacher.eval()
            logger.info("EMA teacher initialized; frozen probability weight=%.3f", float(cfg["frozen_teacher_weight"]))

        judge = None
        if method != "sft":
            judge = Qwen3GuardJudge(
                config["judge"]["path"], batch_size=int(config["judge"]["batch_size"]),
                max_new_tokens=int(config["judge"]["max_new_tokens"]),
            )
        if method == "sft":
            history = train_sft(student, tokenizer, judge, rows, config, run_dir, logger)
        else:
            history = train_onpolicy(student, teacher, tokenizer, judge, rows, config, run_dir, logger)
        if len(history) != int(cfg["steps"]):
            raise RuntimeError("training returned before all requested optimizer steps completed")
        logger.info("training finished; saving adapters and metadata before dev evaluation")
        save_training_checkpoint(student, teacher, run_dir, len(history), completed=True)
        summary = {
            "run_id": run_id, "method": method, "seed": int(config["seed"]),
            "steps": len(history), "final_loss": history[-1]["loss"],
            "mean_seconds_per_step": sum(row["seconds"] for row in history) / len(history),
            "config": config,
        }
        update_run(run_dir, metrics=summary, final_loss=history[-1]["loss"])
        if teacher is not None:
            del teacher, teacher_base
            torch.cuda.empty_cache()
        if not config["eval"].get("run_after_training", True):
            update_run(run_dir, dev={"status": "skipped"})
            return
        update_run(run_dir, dev={"status": "running"})
        try:
            if judge is None:
                judge = Qwen3GuardJudge(
                    config["judge"]["path"], batch_size=int(config["judge"]["batch_size"]),
                    max_new_tokens=int(config["judge"]["max_new_tokens"]),
                )
            dev_path = Path(config["data"]["dev"])
            manifest = json.loads(Path(config["data"]["manifest"]).read_text(encoding="utf-8"))
            dev_info = manifest.get("files", {}).get("dev", {})
            if not dev_info.get("path") or Path(resolve_path(config, dev_info["path"])).resolve() != dev_path.resolve():
                raise ValueError("manifest does not identify the configured dev set")
            if sha256_file(dev_path) != dev_info.get("sha256"):
                raise ValueError("dev artifact hash differs from the manifest")
            dev_rows = read_jsonl(dev_path)
            dev_rows = dev_rows[:int(config["eval"].get("dev_max_rows", len(dev_rows)))]
            dev_metrics = evaluate_dev(student, tokenizer, judge, dev_rows, config)
            dev_metrics["status"] = "completed"
            dev_metrics["dev_sha256"] = dev_info["sha256"]
            write_json(run_dir / "dev_metrics.json", dev_metrics)
            update_run(run_dir, status="completed", dev=dev_metrics)
            logger.info("dev metrics: %s", json.dumps(dev_metrics, sort_keys=True))
        except Exception as error:
            dev_metrics = {"status": "failed", "error": type(error).__name__, "detail": str(error)}
            write_json(run_dir / "dev_metrics.json", dev_metrics)
            update_run(run_dir, status="trained", dev=dev_metrics)
            logger.exception("dev evaluation failed; completed training and saved adapters remain valid")
    except BaseException as error:
        saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if saved.get("training_completed"):
            update_run(run_dir, status="trained", dev={"status": "interrupted", "error": str(error)})
        else:
            update_run(
                run_dir, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                training_completed=False, valid=False,
                error={"type": type(error).__name__, "detail": str(error)},
            )
        logger.exception("run stopped: %s", error)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
