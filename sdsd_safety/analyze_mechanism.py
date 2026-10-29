"""Short mechanism analysis for the safety adaptation.

For a fixed set of held-out harmful prompts, measure:
  * mean log-probability of a fixed refusal prefix at the assistant position;
  * the same with a safe demonstration in context (demo sensitivity / a PCMI
    proxy);
  * per-token KL(student || base) on the prefix, split into early and late
    positions (early-token concentration).

This does not retrain anything; it reads the saved LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from .modeling import chat_prompt, completion_logits, load_base_model, load_tokenizer, teacher_prompt
from .utils import load_config, read_jsonl, resolve_path, write_json

REFUSAL_PREFIX = (
    "I'm sorry, but I can't help with that. I'm not able to provide guidance for that request. "
    "If you have a different question, I'd be happy to help."
)


def prefix_logprobs(model, tokenizer, prompt_texts: list[str], prefix_ids: list[int], batch: int = 4) -> list[float]:
    logits = completion_logits(
        model,
        tokenizer,
        prompt_texts,
        [prefix_ids] * len(prompt_texts),
        micro_batch=batch,
        with_grad=False,
    )
    ids = torch.tensor(prefix_ids, device=logits[0].device)
    values = []
    for row in logits:
        log_probs = torch.log_softmax(row.float(), dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        values.append(float(log_probs.mean()))
    return values


def per_position_kl(student_logits, base_logits) -> list[float]:
    values = []
    for student, base in zip(student_logits, base_logits, strict=True):
        log_s = torch.log_softmax(student.float(), dim=-1)
        log_b = torch.log_softmax(base.float(), dim=-1)
        kl = (log_s.exp() * (log_s - log_b)).sum(-1)
        values.append(kl.mean().item())
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--run", action="append", required=True, help="run_id:adapter_path or run_id:none")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/mechanism.json")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["model"]["path"])
    dev = [row for row in read_jsonl(resolve_path(config, config["data"]["dev"])) if row["kind"] == "harmful"]
    import random

    random.Random(args.seed).shuffle(dev)
    dev = dev[: args.num_prompts]
    prompts = [row["prompt"] for row in dev]
    prefix_ids = tokenizer(REFUSAL_PREFIX, add_special_tokens=False)["input_ids"]
    print(f"refusal prefix tokens: {len(prefix_ids)}")

    results: dict[str, Any] = {}
    for spec in args.run:
        run_id, adapter = spec.split(":", 1)
        base = load_base_model(
            config["model"]["path"],
            dtype=config["model"]["dtype"],
            attn_implementation=config["model"]["attn_implementation"],
        )
        model = base
        if adapter not in {"none", "base", ""}:
            model = PeftModel.from_pretrained(base, adapter, local_files_only=True)
            model = model.to(next(iter(model.parameters())).device)
        model.eval()
        student_prompts = [chat_prompt(tokenizer, prompt) for prompt in prompts]
        demo_prompts = [teacher_prompt(tokenizer, prompt, REFUSAL_PREFIX) for prompt in prompts]
        plain = prefix_logprobs(model, tokenizer, student_prompts, prefix_ids)
        with_demo = prefix_logprobs(model, tokenizer, demo_prompts, prefix_ids)
        student_logits = completion_logits(
            model, tokenizer, student_prompts, [prefix_ids] * len(prompts), micro_batch=4, with_grad=False
        )
        if isinstance(model, PeftModel):
            with torch.no_grad(), model.disable_adapter():
                base_logits = completion_logits(
                    model, tokenizer, student_prompts, [prefix_ids] * len(prompts), micro_batch=4, with_grad=False
                )
        else:
            base_logits = student_logits
        kl = per_position_kl(student_logits, base_logits)
        split = len(prefix_ids) // 2
        early = []
        late = []
        for row_student, row_base in zip(student_logits, base_logits, strict=True):
            log_s = torch.log_softmax(row_student.float(), dim=-1)
            log_b = torch.log_softmax(row_base.float(), dim=-1)
            value = (log_s.exp() * (log_s - log_b)).sum(-1)
            early.append(float(value[:split].mean()))
            late.append(float(value[split:].mean()))
        results[run_id] = {
            "n_prompts": len(prompts),
            "refusal_prefix_logprob": sum(plain) / len(plain),
            "refusal_prefix_logprob_with_demo": sum(with_demo) / len(with_demo),
            "demo_delta": sum(b - a for a, b in zip(plain, with_demo, strict=True)) / len(plain),
            "kl_prefix_mean": sum(kl) / len(kl),
            "kl_early": sum(early) / len(early),
            "kl_late": sum(late) / len(late),
            "kl_early_share": (sum(early) / len(early)) / max((sum(early) / len(early)) + (sum(late) / len(late)), 1e-12),
        }
        print(json.dumps({run_id: results[run_id]}, indent=2), flush=True)
        del model, base
        torch.cuda.empty_cache()
    output = Path(config["project_root"]) / args.output
    write_json(output, results)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
