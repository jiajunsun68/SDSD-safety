"""Add verified refusal targets to the shared, prompt-audited training artifact.

Generated and fallback refusals are both checked against each original prompt.
No rows are removed and no evaluation data are opened. All methods continue to
use the same prompt set and manifest-bound file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .judging import Qwen3GuardJudge
from .modeling import generate_rollouts, load_base_model, load_tokenizer
from .train import OPSA_HARMFUL_PROMPT, response_is_refusal, verified_normal_target
from .utils import load_config, now_cst, read_jsonl, resolve_path, sha256_file, write_json, write_jsonl

FALLBACK_REFUSAL = (
    "I'm sorry, but I can't help with that. "
    "If you have a different question, I'm happy to help."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    path = Path(resolve_path(config, config["data"]["train"])).resolve()
    sft_path = Path(resolve_path(config, config["data"].get("sft_train", config["data"]["train"]))).resolve()
    if path != sft_path:
        raise SystemExit("data.train and data.sft_train must identify the same audited prompt artifact")
    manifest_path = Path(resolve_path(config, config["data"]["manifest"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("prompt_audit", {})
    if audit.get("completed") is not True or audit.get("protocol") != "prompt-intent-v2":
        raise SystemExit("make_sft_targets requires the completed prompt-intent-v2 audit")
    file_info = manifest.get("files", {}).get("train", {})
    if not file_info.get("path") or Path(resolve_path(config, file_info["path"])).resolve() != path:
        raise SystemExit("manifest does not identify the requested training file")
    input_hash = sha256_file(path)
    if file_info.get("sha256") != input_hash:
        raise SystemExit("training data hash differs from the audited manifest")
    for split in ("dev", "test"):
        if config["data"].get(split) and path == Path(resolve_path(config, config["data"][split])).resolve():
            raise SystemExit(f"target generation must not modify the {split} artifact")
    rows = read_jsonl(path)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("training rows must be nonempty and have unique ids")
    for row in rows:
        if row["kind"] == "normal":
            verified_normal_target(row)
        elif row["kind"] != "harmful":
            raise SystemExit("training artifact contains an unaudited kind")
    harmful = [row for row in rows if row["kind"] == "harmful"]
    tokenizer = load_tokenizer(config["model"]["path"])
    model = load_base_model(
        config["model"]["path"],
        dtype=config["model"]["dtype"],
        attn_implementation=config["model"]["attn_implementation"],
    )
    rollouts = generate_rollouts(
        model,
        tokenizer,
        [row["prompt"] for row in harmful],
        num_rollouts=1,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        greedy=True,
        system=OPSA_HARMFUL_PROMPT,
        max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
    )
    responses = [group[0]["text"] for group in rollouts]
    del model
    torch.cuda.empty_cache()
    judge = Qwen3GuardJudge(
        config["judge"]["path"],
        batch_size=int(config["judge"]["batch_size"]),
        max_new_tokens=max(24, int(config["judge"]["max_new_tokens"])),
    )
    judgments = judge.judge_responses([row["prompt"] for row in harmful], responses)
    targets = {}
    self_refusals = 0
    fallbacks = 0
    fallback_rows = []
    for row, response, judgment in zip(harmful, responses, judgments, strict=True):
        if judgment["safety"] == "Safe" and response_is_refusal(response, judgment):
            targets[row["id"]] = (response, "self_refusal")
            self_refusals += 1
        else:
            fallback_rows.append(row)

    fallback_judgments = judge.judge_responses(
        [row["prompt"] for row in fallback_rows], [FALLBACK_REFUSAL] * len(fallback_rows),
    ) if fallback_rows else []
    blocked = []
    for row, judgment in zip(fallback_rows, fallback_judgments, strict=True):
        if judgment["safety"] == "Safe" and response_is_refusal(FALLBACK_REFUSAL, judgment):
            targets[row["id"]] = (FALLBACK_REFUSAL, "verified_fallback_refusal")
            fallbacks += 1
        else:
            blocked.append(row["id"])
    if blocked:
        manifest["targets"] = {
            "completed": False, "status": "blocked", "protocol": "verified-refusal-targets-v2",
            "blocked_ids": blocked, "updated_at": now_cst(),
        }
        write_json(manifest_path, manifest)
        raise SystemExit(f"{len(blocked)} fallback targets failed verification; training artifact was not changed")

    output = []
    for row in rows:
        row = dict(row)
        if row["kind"] == "harmful":
            row["sft_response"], row["sft_source"] = targets[row["id"]]
            row["sft_safety"] = "Safe"
            row["sft_refusal"] = True
        else:
            row["sft_response"] = row["external_response"]
            row["sft_source"] = "audited_external"
            row["sft_safety"] = row["external_safety"]
            row["sft_refusal"] = row["external_refusal"]
        output.append(row)
    write_jsonl(path, output)
    stats = {
        "completed": True,
        "status": "completed",
        "protocol": "verified-refusal-targets-v2",
        "updated_at": now_cst(),
        "rows": len(output),
        "harmful": len(harmful),
        "self_refusals": self_refusals,
        "fallbacks": fallbacks,
        "self_refusal_rate": self_refusals / max(len(harmful), 1),
        "path": str(path),
        "input_sha256": input_hash,
        "model": config["model"]["path"],
        "judge": config["judge"]["path"],
    }
    digest = sha256_file(path)
    for key in ("train", "sft_train"):
        manifest.setdefault("files", {}).setdefault(key, {}).update(path=str(path), sha256=digest)
    manifest["targets"] = stats
    write_json(manifest_path, manifest)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
