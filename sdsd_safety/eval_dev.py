"""Evaluate a checkpoint (or the base model) on the fixed dev set.

Used for the base row and for runs whose in-training dev evaluation was skipped
or OOM.  Writes runs/<run_id>/dev_metrics.json so the summary phase can merge it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from .judging import Qwen3GuardJudge
from .modeling import load_base_model, load_tokenizer
from .train import evaluate_dev
from .utils import load_config, read_jsonl, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["model"]["path"])
    model = load_base_model(
        config["model"]["path"],
        dtype=config["model"]["dtype"],
        attn_implementation=config["model"]["attn_implementation"],
    )
    if args.adapter and args.adapter not in {"none", "base", ""}:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
        model = model.to(next(iter(model.parameters())).device)
    judge = Qwen3GuardJudge(
        config["judge"]["path"],
        batch_size=int(config["judge"]["batch_size"]),
        max_new_tokens=int(config["judge"]["max_new_tokens"]),
    )
    rows = read_jsonl(resolve_path(config, config["data"]["dev"]))
    if args.max_rows:
        rows = rows[: args.max_rows]
    metrics = evaluate_dev(model, tokenizer, judge, rows, config)
    run_dir = Path(resolve_path(config, config["train"]["output_root"])) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "dev_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
