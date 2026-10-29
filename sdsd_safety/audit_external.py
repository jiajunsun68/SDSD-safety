"""Audit and filter external safe responses before building the fixed splits.

PKU-SafeRLHF's `is_response_*_safe` label is not a Qwen3Guard label.  A pilot
found that only ~56% of the "safe" PKU responses actually pass Qwen3Guard, which
would corrupt both SFT targets and SDSD fallback demonstrations.  This script
judges the whole harmful candidate pool with Qwen3Guard, keeps only Safe external
responses, rebuilds the same fixed splits, and audits the selected normal
targets.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .judging import Qwen3GuardJudge
from .metrics import is_refusal
from .prepare_data import (
    dedup,
    load_alpaca,
    load_eval_prompts,
    load_pku,
    normalize,
    take,
    valid_harmful,
    valid_normal,
)
from .utils import load_config, now_cst, resolve_path, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    min_response_chars = int(config["data"]["min_response_chars"])
    rng = random.Random(seed)

    pku_rows = dedup(
        [
            row
            for row in load_pku(resolve_path(config, config["sources"]["pku_train_arrow"]))
            if valid_harmful(row, min_response_chars)
        ]
    )
    alpaca_rows = dedup(
        [
            row
            for row in load_alpaca(resolve_path(config, config["sources"]["alpaca_json"]))
            if valid_normal(row, min_response_chars)
        ]
    )
    eval_prompts = load_eval_prompts(config)
    pku_rows = [row for row in pku_rows if normalize(row["prompt"]) not in eval_prompts]
    alpaca_rows = [row for row in alpaca_rows if normalize(row["prompt"]) not in eval_prompts]

    judge = Qwen3GuardJudge(
        config["judge"]["path"],
        batch_size=args.batch_size,
        max_new_tokens=int(config["judge"]["max_new_tokens"]),
    )
    prompts = [row["prompt"] for row in pku_rows]
    responses = [row["external_response"] for row in pku_rows]
    judgments = []
    for start in range(0, len(prompts), args.batch_size):
        judgments.extend(judge.judge_responses(prompts[start : start + args.batch_size], responses[start : start + args.batch_size]))
        print(f"judged harmful external {len(judgments)}/{len(prompts)}", flush=True)
    audit_rows = []
    kept_harmful = []
    for row, judgment in zip(pku_rows, judgments, strict=True):
        audit_rows.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "safety": judgment["safety"],
                "refusal": judgment["refusal"],
            }
        )
        if judgment["safety"] == "Safe":
            row = dict(row)
            row["external_safety"] = "Safe"
            kept_harmful.append(row)
    write_jsonl(Path(config["project_root"]) / "data" / "external_audit_harmful.jsonl", audit_rows)

    split_cfg = config["split"]
    harmful_train = take(kept_harmful, split_cfg["harmful_train"], rng, set(), "train")
    forbidden = {normalize(row["prompt"]) for row in harmful_train}
    harmful_dev = take(kept_harmful, split_cfg["harmful_dev"], rng, forbidden, "dev")
    forbidden |= {normalize(row["prompt"]) for row in harmful_dev}
    harmful_test = take(kept_harmful, split_cfg["harmful_test"], rng, forbidden, "test")

    forbidden = {normalize(row["prompt"]) for row in harmful_train + harmful_dev + harmful_test}
    normal_candidates = take(
        alpaca_rows,
        split_cfg["normal_train"] + split_cfg["normal_dev"] + split_cfg["normal_test"] + 300,
        random.Random(seed + 7),
        forbidden,
        "candidate",
    )
    normal_judgments = []
    for start in range(0, len(normal_candidates), args.batch_size):
        chunk = normal_candidates[start : start + args.batch_size]
        normal_judgments.extend(
            judge.judge_responses(
                [row["prompt"] for row in chunk],
                [row["external_response"] for row in chunk],
            )
        )
        print(f"judged normal external {len(normal_judgments)}/{len(normal_candidates)}", flush=True)
    safe_normal = []
    unsafe_normal = 0
    for row, judgment in zip(normal_candidates, normal_judgments, strict=True):
        audit_rows.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "safety": judgment["safety"],
                "refusal": judgment["refusal"],
                "text_refusal": is_refusal(row["external_response"]),
            }
        )
        if judgment["safety"] == "Safe" and not is_refusal(row["external_response"]):
            row = dict(row)
            row["external_safety"] = "Safe"
            safe_normal.append(row)
        else:
            unsafe_normal += 1
    write_jsonl(Path(config["project_root"]) / "data" / "external_audit.jsonl", audit_rows)

    normal_rng = random.Random(seed + 11)
    normal_train = take(safe_normal, split_cfg["normal_train"], normal_rng, set(), "train")
    normal_forbidden = {normalize(row["prompt"]) for row in normal_train}
    normal_dev = take(safe_normal, split_cfg["normal_dev"], normal_rng, normal_forbidden, "dev")
    normal_forbidden |= {normalize(row["prompt"]) for row in normal_dev}
    normal_test = take(safe_normal, split_cfg["normal_test"], normal_rng, normal_forbidden, "test")

    train = harmful_train + normal_train
    dev = harmful_dev + normal_dev
    test = harmful_test + normal_test
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    root = Path(config["project_root"])
    paths = {
        "train": root / config["data"]["train"],
        "dev": root / config["data"]["dev"],
        "test": root / config["data"]["test"],
    }
    write_jsonl(paths["train"], train)
    write_jsonl(paths["dev"], dev)
    write_jsonl(paths["test"], test)
    manifest = {
        "created_at": now_cst(),
        "seed": seed,
        "sources": {
            "pku_train_arrow": config["sources"]["pku_train_arrow"],
            "alpaca_json": config["sources"]["alpaca_json"],
        },
        "external_audit": {
            "judge": config["judge"]["path"],
            "harmful_candidates": len(pku_rows),
            "harmful_kept_safe": len(kept_harmful),
            "harmful_keep_rate": len(kept_harmful) / max(len(pku_rows), 1),
            "normal_checked": len(selected_normal),
            "normal_unsafe_or_refusal": unsafe_normal,
        },
        "counts": {
            "train": {"total": len(train), "harmful": len(harmful_train), "normal": len(normal_train)},
            "dev": {"total": len(dev), "harmful": len(harmful_dev), "normal": len(normal_dev)},
            "test": {"total": len(test), "harmful": len(harmful_test), "normal": len(normal_test)},
        },
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_json(root / "data" / "manifest.json", manifest)
    print(json.dumps(manifest["external_audit"], indent=2))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
