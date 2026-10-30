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
from .train import evaluate_dev, update_run
from .utils import load_config, read_jsonl, resolve_path, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(resolve_path(config, config["train"]["output_root"])) / args.run_id
    run_path = run_dir / "run.json"
    metadata = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
    is_base = args.adapter in {"none", "base"}
    if not is_base:
        if not metadata or metadata.get("protocol_id") != "sdsd-safety-training-v2":
            raise SystemExit("dev evaluation requires repaired-run metadata, or --adapter none for the base model")
        if not metadata.get("training_completed") or not metadata.get("valid"):
            raise SystemExit("refusing to evaluate an incomplete or unverified training run as final")
        config = metadata["config"]
        checkpoint = Path(metadata["checkpoint"]).resolve()
        if args.adapter and Path(args.adapter).resolve() != checkpoint:
            raise SystemExit("adapter argument differs from the saved run checkpoint")
        args.adapter = str(checkpoint)
    manifest_path = Path(resolve_path(config, config["data"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev_path = Path(resolve_path(config, config["data"]["dev"])).resolve()
    dev_info = manifest.get("files", {}).get("dev", {})
    if not dev_info.get("path") or Path(resolve_path(config, dev_info["path"])).resolve() != dev_path:
        raise SystemExit("manifest does not identify the selected dev set")
    if sha256_file(dev_path) != dev_info.get("sha256"):
        raise SystemExit("dev artifact hash differs from the manifest")
    if metadata and not is_base:
        update_run(run_dir, dev={"status": "running"})
    try:
        metrics = run_dev_evaluation(config, args.adapter, args.max_rows, dev_path)
    except BaseException as error:
        failure = {"status": "failed", "error": type(error).__name__, "detail": str(error)}
        write_json(run_dir / "dev_metrics.json", failure)
        if metadata and not is_base:
            update_run(run_dir, status="trained", dev=failure)
        raise
    metrics.update(status="completed", dev_sha256=dev_info["sha256"])
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "dev_metrics.json", metrics)
    if metadata and not is_base:
        update_run(run_dir, status="completed", dev=metrics)
    print(json.dumps(metrics, indent=2))


def run_dev_evaluation(config, adapter, max_rows, dev_path):
    tokenizer = load_tokenizer(config["model"]["path"])
    model = load_base_model(
        config["model"]["path"],
        dtype=config["model"]["dtype"],
        attn_implementation=config["model"]["attn_implementation"],
    )
    if adapter and adapter not in {"none", "base", ""}:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        model = model.to(next(iter(model.parameters())).device)
    judge = Qwen3GuardJudge(
        config["judge"]["path"],
        batch_size=int(config["judge"]["batch_size"]),
        max_new_tokens=int(config["judge"]["max_new_tokens"]),
    )
    rows = read_jsonl(dev_path)
    if max_rows:
        rows = rows[:max_rows]
    return evaluate_dev(model, tokenizer, judge, rows, config)


if __name__ == "__main__":
    main()
