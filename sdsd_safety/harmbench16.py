"""HarmBench repeated-sampling safety metric (ASR@1 and ASR@16).

One greedy HarmBench sample can saturate for an already-safe instruct model.
This script samples 16 responses per behavior at temperature 1.0, labels every
response with the official HarmBench classifier, and reports both mean ASR@1
and the fraction of behaviors with at least one harmful response (ASR@16).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate import HARMBENCH_CLASSIFIER_PROMPT, harmbench_rows, raw_dir
from .metrics import parse_harmbench_label
from .modeling import generate_rollouts, load_base_model, load_tokenizer
from .utils import load_config, read_jsonl, resolve_path, write_json, write_jsonl


def generate(config: dict[str, Any], run_id: str, adapter: str | None, samples: int, max_behaviors: int | None) -> None:
    tokenizer = load_tokenizer(config["model"]["path"])
    model = load_base_model(
        config["model"]["path"],
        dtype=config["model"]["dtype"],
        attn_implementation=config["model"]["attn_implementation"],
    )
    if adapter and adapter not in {"none", "base", ""}:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        model = model.to(next(iter(model.parameters())).device)
    rows = harmbench_rows(config)
    if max_behaviors:
        import random

        random.Random(int(config["seed"])).shuffle(rows)
        rows = rows[:max_behaviors]
    rollouts = generate_rollouts(
        model,
        tokenizer,
        [row["prompt"] for row in rows],
        num_rollouts=samples,
        max_new_tokens=int(config["eval"]["max_new_tokens"]),
        temperature=float(config["train"]["temperature"]),
        top_p=float(config["train"]["top_p"]),
        batch_size=4,
        greedy=False,
        max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
    )
    output = [
        {**row, "responses": [sample["text"] for sample in group]}
        for row, group in zip(rows, rollouts, strict=True)
    ]
    write_jsonl(raw_dir(config, run_id) / f"harmbench{samples}.jsonl", output)
    print(f"[{run_id}] generated {len(output)} x {samples}", flush=True)
    del model
    torch.cuda.empty_cache()


def judge(config: dict[str, Any], run_ids: list[str], samples: int) -> None:
    classifier_path = resolve_path(config, config["sources"]["harmbench_classifier"])
    tokenizer = AutoTokenizer.from_pretrained(classifier_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        classifier_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for run_id in run_ids:
        rows = read_jsonl(raw_dir(config, run_id) / f"harmbench{samples}.jsonl")
        flat = [(index, response) for index, row in enumerate(rows) for response in row["responses"]]
        prompts = [
            HARMBENCH_CLASSIFIER_PROMPT.format(behavior=rows[index]["prompt"], generation=response)
            for index, response in flat
        ]
        labels: list[int | None] = []
        with torch.inference_mode():
            for start in range(0, len(prompts), 8):
                batch = tokenizer(prompts[start : start + 8], padding=True, return_tensors="pt").to(model.device)
                generated = model.generate(
                    **batch,
                    do_sample=False,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
                texts = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
                labels.extend(parse_harmbench_label(text) for text in texts)
        per_row = [[] for _ in rows]
        for (index, _), label in zip(flat, labels, strict=True):
            per_row[index].append(label)
        all_labels = [int(label) for label in labels if label is not None]
        any_harmful = [any(label == 1 for label in row if label is not None) for row in per_row]
        rng = np.random.default_rng(int(config["seed"]))
        sampled = rng.choice(all_labels, size=(10000, len(all_labels)), replace=True).mean(axis=1)
        summary = {
            "benchmark": f"HarmBench@{samples}",
            "n_behaviors": len(rows),
            "n_generations": len(labels),
            "asr_at_1": float(np.mean(all_labels)) if all_labels else None,
            "asr_at_1_ci95_low": float(np.quantile(sampled, 0.025)),
            "asr_at_1_ci95_high": float(np.quantile(sampled, 0.975)),
            "asr_at_k": float(np.mean(any_harmful)),
            "k": samples,
            "parse_failure_rate": 1.0 - len(all_labels) / max(len(rows) * samples, 1),
        }
        write_json(raw_dir(config, run_id) / f"harmbench{samples}_summary.json", summary)
        print(f"[{run_id}] {json.dumps(summary, sort_keys=True)}", flush=True)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--run", action="append", required=True, help="run_id:adapter_path or run_id:none")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-behaviors", type=int, default=None)
    parser.add_argument("--phase", choices=["generate", "judge", "all"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    specs = [spec.split(":", 1) for spec in args.run]
    if args.phase in {"generate", "all"}:
        for run_id, adapter in specs:
            generate(config, run_id, adapter, args.samples, args.max_behaviors)
    if args.phase in {"judge", "all"}:
        judge(config, [run_id for run_id, _ in specs], args.samples)


if __name__ == "__main__":
    main()
