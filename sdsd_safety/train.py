"""Unified trainer for SFT, GRPO, OPSA and SDSD-Safety.

All four methods share the model, data split, LoRA parameterization, decoding
settings and evaluation code.  Only the objective differs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .judging import Qwen3GuardJudge, judge_summary
from .losses import grpo_loss, sft_loss, token_kl
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
    lora_parameters,
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def stratified_steps(rows: list[dict[str, Any]], batch: int, steps: int, seed: int):
    """Yield batches preserving the harmful/normal ratio of the training split."""
    harmful = [row for row in rows if row["kind"] == "harmful"]
    normal = [row for row in rows if row["kind"] == "normal"]
    n_harmful = max(1, round(batch * len(harmful) / len(rows)))
    n_normal = batch - n_harmful
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


def select_demos(
    row: dict[str, Any],
    samples: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[str], int]:
    """Return (demonstrations, number taken from the student's own rollouts)."""
    min_tokens = int(config["data"].get("min_demo_tokens", 4))
    qualified = []
    for sample, judgment in zip(samples, judgments, strict=True):
        safe = judgment.get("safety") == "Safe"
        if row["kind"] == "normal":
            safe = safe and not is_refusal(sample["text"]) and len(sample["token_ids"]) >= min_tokens
        if safe:
            qualified.append(sample["text"])
    if row["kind"] == "harmful" and config["train"].get("demo_preference") == "refusal":
        refusals = [text for text in qualified if is_refusal(text)]
        if refusals:
            qualified = refusals
    rng.shuffle(qualified)
    external = [row["external_response"]]
    num_demos = int(config["train"].get("num_demos", 1))
    source = config["train"].get("demo_source", "self")
    if source == "external":
        demos = external * num_demos
        return demos, 0
    if source == "external_refusal":
        demo = row.get("sft_response") or row["external_response"]
        return [demo] * num_demos, 0
    demos = qualified[:num_demos]
    used_self = len(demos)
    if len(demos) < num_demos:
        demos += external * (num_demos - len(demos))
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
        is_refusal(response)
        for row, response in zip(rows, responses, strict=True)
        if row["kind"] == "normal"
    ]
    return {
        "harmful": judge_summary(harmful),
        "normal": judge_summary(normal),
        "normal_refusal_rate": sum(normal_refusal) / max(len(normal_refusal), 1),
        "n": len(rows),
    }


def save_run(
    run_dir: Path,
    student,
    config: dict[str, Any],
    metrics: dict[str, Any],
    command: str,
) -> None:
    checkpoint = run_dir / "checkpoint"
    student.save_pretrained(checkpoint)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(
        run_dir / "run.json",
        {
            "created_at": now_cst(),
            "git_commit": git_commit(),
            "command": command,
            "metrics": metrics,
            "checkpoint": str(checkpoint),
            "trainable_parameters": int(sum(p.numel() for p in student.parameters() if p.requires_grad)),
        },
    )


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
    for step, batch_rows in enumerate(stratified_steps(rows, batch, int(cfg["steps"]), int(config["seed"])), start=1):
        started = time.time()
        student.train()
        optimizer.zero_grad(set_to_none=True)
        encoded = [
            encode_sft_pair(tokenizer, row["prompt"], row.get("sft_response") or row["external_response"], max_length)
            for row in batch_rows
        ]
        num_micro = math.ceil(len(encoded) / micro)
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
            (loss / num_micro).backward()
            loss_value += float(loss.item())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], float(cfg["grad_clip"])
        )
        optimizer.step()
        record = {
            "step": step,
            "loss": loss_value / num_micro,
            "grad_norm": float(grad_norm),
            "seconds": time.time() - started,
        }
        history.append(record)
        if step % int(cfg["log_every"]) == 0:
            logger.info(
                "step %d/%d loss=%.4f grad=%.3f sec=%.1f",
                step,
                int(cfg["steps"]),
                record["loss"],
                record["grad_norm"],
                record["seconds"],
            )
            append_jsonl(run_dir / "metrics.jsonl", record)
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
    history = []
    for step, batch_rows in enumerate(stratified_steps(rows, batch, int(cfg["steps"]), int(config["seed"])), start=1):
        started = time.time()
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
        flat_rows, flat_samples, flat_judgments = [], [], []
        flat_demos: list[list[str]] = []
        flat_demo_self: list[int] = []
        for row, samples in zip(batch_rows, rollouts, strict=True):
            judgments = judge.judge_responses([row["prompt"]] * n, [sample["text"] for sample in samples])
            for sample, judgment in zip(samples, judgments, strict=True):
                flat_rows.append(row)
                flat_samples.append(sample)
                flat_judgments.append(judgment)
            if method == "sdsd":
                for _ in range(n):
                    demos, used_self = select_demos(row, samples, judgments, config, rng)
                    flat_demos.append(demos)
                    flat_demo_self.append(used_self)

        safe_flags = [judgment.get("safety") == "Safe" for judgment in flat_judgments]
        refusal_flags = [is_refusal(sample["text"]) for sample in flat_samples]
        completion_ids = [sample["token_ids"] for sample in flat_samples]
        student_prompts = [chat_prompt(tokenizer, row["prompt"]) for row in flat_rows]

        student.train()
        optimizer.zero_grad(set_to_none=True)
        num_micro = math.ceil(len(flat_rows) / micro)
        loss_sum = 0.0
        extra: dict[str, float] = {}
        demo_self = 0
        demo_total = 0
        advantages: list[float] = []
        old_logprobs: list[torch.Tensor] = []
        ref_logprobs: list[torch.Tensor] = []

        if method == "grpo":
            margins = judge.score_responses(
                [row["prompt"] for row in flat_rows],
                [sample["text"] for sample in flat_samples],
            )
            rewards = []
            for row, margin, refusal in zip(flat_rows, margins, refusal_flags, strict=True):
                if row["kind"] == "harmful":
                    rewards.append(float(margin) + (4.0 if refusal else 0.0))
                else:
                    rewards.append(float(margin) - (8.0 if refusal else 0.0))
            for start in range(0, len(flat_rows), n):
                group = rewards[start : start + n]
                mean = sum(group) / len(group)
                variance = sum((value - mean) ** 2 for value in group) / len(group)
                std = math.sqrt(variance + 1e-6)
                advantages.extend((value - mean) / std for value in group)
            student.eval()
            with torch.no_grad():
                old_logits = completion_logits(
                    student, tokenizer, student_prompts, completion_ids,
                    with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                )
                from .losses import gathered_logprobs

                old_logprobs = gathered_logprobs(old_logits, completion_ids)
                with student.disable_adapter():
                    ref_logits = completion_logits(
                        student, tokenizer, student_prompts, completion_ids,
                        with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                    )
                ref_logprobs = gathered_logprobs(ref_logits, completion_ids)
            student.train()

        for start in range(0, len(flat_rows), micro):
            chunk_rows = flat_rows[start : start + micro]
            chunk_ids = completion_ids[start : start + micro]
            chunk_prompts = student_prompts[start : start + micro]
            logits = completion_logits(
                student, tokenizer, chunk_prompts, chunk_ids,
                micro_batch=min(4, len(chunk_rows)), with_grad=True,
                max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
            )
            if method == "grpo":
                loss, stats = grpo_loss(
                    logits,
                    chunk_ids,
                    old_logprobs[start : start + micro],
                    ref_logprobs[start : start + micro],
                    advantages[start : start + micro],
                    ref_coef=float(cfg["ref_kl_coef"]),
                )
                extra["clip_fraction"] = extra.get("clip_fraction", 0.0) + stats["clip_fraction"] / num_micro
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
                student.train()
                loss = token_kl(
                    logits, teacher_logits, direction="mixed",
                    mixed_weight=float(cfg["opsa_mixed_kl_weight"]),
                )
            elif method == "sdsd":
                demo_lists = flat_demos[start : start + micro]
                demo_self += sum(flat_demo_self[start : start + micro])
                demo_total += sum(len(demos) for demos in demo_lists)
                per_demo = []
                for demo_index in range(len(demo_lists[0])):
                    teacher_prompts = [
                        teacher_prompt(tokenizer, row["prompt"], demos[demo_index])
                        for row, demos in zip(chunk_rows, demo_lists, strict=True)
                    ]
                    with torch.no_grad():
                        if teacher is not None:
                            teacher_logits = completion_logits(
                                teacher, tokenizer, teacher_prompts, chunk_ids,
                                with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                            )
                        else:
                            with student.disable_adapter():
                                teacher_logits = completion_logits(
                                    student, tokenizer, teacher_prompts, chunk_ids,
                                    with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                                )
                    per_demo.append(token_kl(logits, teacher_logits, direction="reverse"))
                loss = torch.stack(per_demo).mean()
                student.eval()
                with torch.no_grad(), student.disable_adapter():
                    ref_logits = completion_logits(
                        student, tokenizer, chunk_prompts, chunk_ids,
                        with_grad=False, max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
                    )
                student.train()
                loss = loss + float(cfg["ref_kl_coef"]) * token_kl(logits, ref_logits, direction="reverse")
            else:
                raise ValueError(f"unknown method {method}")
            (loss / num_micro).backward()
            loss_sum += float(loss.item())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], float(cfg["grad_clip"])
        )
        optimizer.step()
        if method == "sdsd" and teacher is not None:
            ema_lora(student, teacher, decay=1.0 - float(cfg["teacher_update_rate"]))

        harmful_idx = [i for i, row in enumerate(flat_rows) if row["kind"] == "harmful"]
        normal_idx = [i for i, row in enumerate(flat_rows) if row["kind"] == "normal"]
        record = {
            "step": step,
            "loss": loss_sum / num_micro,
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
        history.append(record)
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
            append_jsonl(run_dir / "metrics.jsonl", record)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    cfg = config["train"]
    method = cfg["method"]
    if args.run_id:
        run_id = args.run_id
    elif method == "sdsd":
        run_id = f"sdsd_{cfg.get('demo_source', 'self')}_k{cfg.get('num_demos', 1)}_seed{config['seed']}"
    else:
        run_id = f"{method}_seed{config['seed']}"
    run_dir = Path(resolve_path(config, cfg["output_root"])) / run_id
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"run directory exists: {run_dir} (use --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "train.log", name=f"sdsd.{run_id}")
    logger.info("run_id=%s method=%s config=%s", run_id, method, json.dumps(cfg, sort_keys=True))
    logger.info("git_commit=%s", git_commit())
    free, total = gpu_memory_gib()
    logger.info("gpu free=%.1f GiB total=%.1f GiB", free, total)
    if free < 25:
        raise SystemExit(f"not enough free GPU memory: {free:.1f} GiB")

    set_seed(int(config["seed"]))
    tokenizer = load_tokenizer(config["model"]["path"])
    base = load_base_model(
        config["model"]["path"],
        dtype=config["model"]["dtype"],
        attn_implementation=config["model"]["attn_implementation"],
    )
    student = attach_lora(base, config["model"]["lora"])
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("trainable parameters: %d (%.2fM)", trainable, trainable / 1e6)

    teacher = None
    if method == "sdsd" and cfg.get("teacher_mode", "ema") == "ema":
        teacher_base = load_base_model(
            config["model"]["path"],
            dtype=config["model"]["dtype"],
            attn_implementation=config["model"]["attn_implementation"],
        )
        teacher = attach_lora(teacher_base, config["model"]["lora"])
        copy_lora(student, teacher)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        logger.info("EMA teacher initialized from student")

    judge = None
    if method != "sft":
        judge = Qwen3GuardJudge(
            config["judge"]["path"],
            batch_size=int(config["judge"]["batch_size"]),
            max_new_tokens=int(config["judge"]["max_new_tokens"]),
        )
    train_key = "sft_train" if method == "sft" and config["data"].get("sft_train") else "train"
    rows = read_jsonl(resolve_path(config, config["data"][train_key]))
    logger.info("training rows: %d", len(rows))

    if method == "sft":
        history = train_sft(student, tokenizer, judge, rows, config, run_dir, logger)
    else:
        history = train_onpolicy(student, teacher, tokenizer, judge, rows, config, run_dir, logger)

    logger.info("training finished; saving adapter before dev evaluation")
    student.save_pretrained(run_dir / "checkpoint")
    if teacher is not None:
        teacher.save_pretrained(run_dir / "teacher_checkpoint")
    if judge is None:
        judge = Qwen3GuardJudge(
            config["judge"]["path"],
            batch_size=int(config["judge"]["batch_size"]),
            max_new_tokens=int(config["judge"]["max_new_tokens"]),
        )
    dev_rows = read_jsonl(resolve_path(config, config["data"]["dev"]))
    dev_limit = int(config["eval"].get("dev_max_rows", len(dev_rows)))
    dev_rows = dev_rows[:dev_limit]
    try:
        dev_metrics = evaluate_dev(student, tokenizer, judge, dev_rows, config)
        logger.info("dev metrics: %s", json.dumps(dev_metrics, sort_keys=True))
    except torch.cuda.OutOfMemoryError as error:
        logger.error("dev evaluation OOM; checkpoint is already saved: %s", error)
        torch.cuda.empty_cache()
        dev_metrics = {"error": "cuda_oom", "detail": str(error)}
    summary = {
        "run_id": run_id,
        "method": method,
        "seed": int(config["seed"]),
        "steps": len(history),
        "final_loss": history[-1]["loss"] if history else None,
        "mean_seconds_per_step": sum(row["seconds"] for row in history) / max(len(history), 1),
        "dev": dev_metrics,
        "config": config,
    }
    save_run(run_dir, student, config, summary, " ".join([__file__, *__import__("sys").argv[1:]]))
    write_json(run_dir / "dev_metrics.json", dev_metrics)
    logger.info("saved checkpoint and summary to %s", run_dir)


if __name__ == "__main__":
    main()
