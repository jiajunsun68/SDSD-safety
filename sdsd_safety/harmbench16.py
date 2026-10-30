"""Fixed-prompt repeated sampling on the official HarmBench standard-200 pool.

This is a fixed decoding/attack-budget evaluation, not an adaptive jailbreak.
For each behavior we draw the same declared number of independent samples and
report the combinatorial ASR@k for k in {1, 4, 16}. Missing judgments are not
silently treated as safe or harmful: complete-case points and an explicit
all-prompt envelope are stored together. Prompt clusters, never flattened
responses, are the bootstrap unit.

The protocol-written manifest separates the frozen training identity (adapter,
run metadata, data) from this evaluation identity (behaviors, attack wrapper,
decoding, judge, code and output hashes). Outputs live in a protocol-specific
directory and never overwrite the greedy v2 evaluation files.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from .evaluate import (
    HARMBENCH_TEMPLATE_SOURCE, classify_harmbench_rows, file_hash, git_commit,
    git_worktree_status, harmbench_rows, load_harmbench_classifier,
    load_model_for_eval, model_identity, object_hash, source_path,
    training_provenance,
)
from .metrics import ASR_ESTIMATOR_ID, ASR_SAMPLING_PROTOCOL_ID, asr_at_k_table
from .utils import (
    load_config, now_cst, read_jsonl, resolve_path, set_seed, write_json,
    write_jsonl,
)

SAMPLING_PROTOCOL_ID = ASR_SAMPLING_PROTOCOL_ID
SAMPLING_DIR = Path("results") / "raw_sampling" / "asr-sample-v3"
SAMPLING_CODE_FILES = (
    "harmbench16.py", "metrics.py", "evaluate.py", "modeling.py", "judging.py", "utils.py",
)
MANIFEST_NAME = "sampling_manifest.json"


def sampling_dir(config: dict[str, Any], run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    directory = Path(config["project_root"]) / SAMPLING_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sampling_code_hashes() -> dict[str, str]:
    package = Path(__file__).parent
    return {name: file_hash(package / name) for name in SAMPLING_CODE_FILES}


def _behavior_identity(rows: list[dict[str, Any]], attack_mode: str = "direct") -> str:
    if attack_mode != "direct":
        raise ValueError("unsupported attack mode")
    return object_hash([{"behavior_id": row["id"], "behavior": row["prompt"],
                         "attack_prompt": row["prompt"]} for row in rows])


def _adapter_identity(adapter: str | None) -> dict[str, Any] | None:
    return model_identity(adapter, adapter=True) if adapter and adapter not in {"none", "base", ""} else None


def _load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing sampling manifest: {path}")
    return json.loads(path.read_text())


def _validate_generated(directory: Path, manifest: dict[str, Any], samples: int,
                        expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_path = directory / "harmbench16.jsonl"
    rows = read_jsonl(raw_path)
    if len(rows) != len(expected) or manifest.get("n") != len(expected):
        raise ValueError("saved sampling rows do not match the manifest behavior count")
    if manifest.get("raw_sha256") != file_hash(raw_path):
        raise ValueError("sampling raw output hash differs from the generation manifest")
    expected_by_id = {row["id"]: row for row in expected}
    seen = set()
    for row in rows:
        behavior_id = row.get("behavior_id")
        if behavior_id not in expected_by_id or behavior_id in seen:
            raise ValueError(f"unknown or duplicate behavior ID: {behavior_id}")
        seen.add(behavior_id)
        original = expected_by_id[behavior_id]
        if row.get("behavior") != original["prompt"]:
            raise ValueError(f"behavior text changed for {behavior_id}")
        if row.get("attack_prompt") != original["prompt"] or row.get("attack_mode") != "direct":
            raise ValueError(f"attack wrapper/behavior separation is invalid for {behavior_id}")
        if len(row.get("responses", [])) != samples:
            raise ValueError(f"expected {samples} responses for {behavior_id}")
        if any(not isinstance(response, str) for response in row["responses"]):
            raise ValueError(f"non-string response for {behavior_id}")
    if seen != set(expected_by_id):
        raise ValueError("saved sampling rows are missing official behaviors")
    return rows


def _selection(config: dict[str, Any], max_behaviors: int | None) -> list[dict[str, Any]]:
    rows = harmbench_rows(config)
    if max_behaviors is not None:
        if max_behaviors < 1:
            raise ValueError("max_behaviors must be positive")
        random.Random(int(config["seed"]) + 991).shuffle(rows)
        rows = rows[:max_behaviors]
    return rows


def generate(config: dict[str, Any], run_id: str, adapter: str | None, samples: int,
             max_behaviors: int | None) -> None:
    if samples < 1:
        raise ValueError("samples must be positive")
    rows = _selection(config, max_behaviors)
    directory = sampling_dir(config, run_id)
    manifest_path = directory / MANIFEST_NAME
    raw_path = directory / "harmbench16.jsonl"
    if raw_path.exists() or manifest_path.exists():
        try:
            manifest = _load_manifest(directory)
            if (manifest.get("status") == "completed" and manifest.get("valid") is True
                    and manifest.get("raw_sha256") == file_hash(raw_path)
                    and manifest.get("samples") == samples and manifest.get("n") == len(rows)):
                print(f"[{run_id}] completed sampling already present; not overwriting", flush=True)
                return
        except Exception:
            pass
        raise FileExistsError(f"sampling output exists but is not a complete verified artifact: {directory}")

    import torch
    from .modeling import chat_prompt, generate_rollouts

    decoding = {
        "do_sample": True, "samples": samples,
        "max_new_tokens": int(config["eval"]["max_new_tokens"]),
        "temperature": float(config["eval"].get("sampling_temperature", 1.0)),
        "top_p": float(config["eval"].get("sampling_top_p", 1.0)), "top_k": 0,
        "batch_size": 4, "max_prompt_tokens": int(config["data"]["max_prompt_tokens"]),
        "seed": int(config["seed"]),
    }
    set_seed(int(config["seed"]))
    model, tokenizer = load_model_for_eval(config, adapter)
    source = source_path(config, "harmbench")
    code = sampling_code_hashes()
    adapter_id = _adapter_identity(adapter)
    training = training_provenance(config, run_id, adapter)
    manifest: dict[str, Any] = {
        "protocol_id": SAMPLING_PROTOCOL_ID, "status": "generating", "valid": False,
        "run_id": run_id, "samples": samples, "n": len(rows),
        "sampling_seed": int(config["seed"]), "started_at": now_cst(),
        "git_commit": git_commit(), **git_worktree_status(),
        "command": shlex.join([sys.executable, *sys.argv]), "argv": [sys.executable, *sys.argv],
        "code_sha256": code,
        "config": config, "config_sha256": object_hash(config),
        "model_identity": model_identity(resolve_path(config, config["model"]["path"])),
        "adapter_identity": adapter_id,
        "training_identity": training,
        "training_identity_sha256": object_hash(training),
        "behavior_source_path": str(source), "behavior_source_sha256": file_hash(source),
        "behavior_sha256": _behavior_identity(rows),
        "attack_mode": "direct", "attack_prompt_sha256": _behavior_identity(rows),
        "decoding": decoding, "generation_config": model.generation_config.to_dict(),
        "chat_template_sha256": object_hash(tokenizer.chat_template),
        "judge": None, "raw_sha256": None, "judged_sha256": None,
        "prompt_counts_sha256": None, "summary_sha256": None,
        "evaluation_identity": None,
    }
    write_json(manifest_path, manifest)
    started = time.time()
    rollouts = generate_rollouts(
        model, tokenizer, [row["prompt"] for row in rows], num_rollouts=samples,
        max_new_tokens=decoding["max_new_tokens"], temperature=decoding["temperature"],
        top_p=decoding["top_p"], batch_size=decoding["batch_size"], greedy=False,
        max_prompt_tokens=decoding["max_prompt_tokens"],
    )
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    output = []
    for row, group in zip(rows, rollouts, strict=True):
        if len(group) != samples:
            raise RuntimeError(f"expected {samples} samples for {row['id']}, got {len(group)}")
        output.append({
            "behavior_id": row["id"], "behavior": row["prompt"],
            "attack_prompt": row["prompt"], "attack_mode": "direct",
            "category": row["category"], "functional_category": row["functional_category"],
            "tags": row["tags"],
            "responses": [sample["text"] for sample in group],
            "response_token_counts": [len(sample["token_ids"]) for sample in group],
            "finish_reasons": [
                "eos" if sample["token_ids"] and sample["token_ids"][-1] in eos_ids
                else "length" if len(sample["token_ids"]) >= decoding["max_new_tokens"] else "other"
                for sample in group
            ],
        })
    write_jsonl(raw_path, output)
    manifest.update(status="generated", raw_sha256=file_hash(raw_path),
                    completed_generation_at=now_cst(), seconds=time.time() - started)
    write_json(manifest_path, manifest)
    print(f"[{run_id}] generated {len(output)} x {samples}, seed={config['seed']}, seconds={time.time()-started:.1f}",
          flush=True)
    del model
    torch.cuda.empty_cache()


def classifier_inputs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build judge inputs from the original behavior, never the attack text."""
    output = []
    for row in rows:
        behavior = row.get("behavior")
        attack_prompt = row.get("attack_prompt")
        if not isinstance(behavior, str) or not behavior.strip() or not isinstance(attack_prompt, str):
            raise ValueError("sampling rows must retain separate behavior and attack_prompt fields")
        for response in row["responses"]:
            output.append({"behavior": behavior, "response": response})
    return output


def _judge_summary(per_prompt: list[list[bool | None]], samples: int, seed: int) -> dict[str, Any]:
    ks = sorted({1, min(4, samples), samples})
    endpoints = asr_at_k_table(per_prompt, ks, seed)
    counts = {
        "n_behaviors": len(per_prompt),
        "n_generations": len(per_prompt) * samples,
        "n_valid_labels": sum(label is not None for row in per_prompt for label in row),
        "n_unknown_labels": sum(label is None for row in per_prompt for label in row),
        "n_completely_judged_behaviors": sum(all(label is not None for label in row) for row in per_prompt),
    }
    summary: dict[str, Any] = {
        "benchmark": f"HarmBench-standard200@{samples}",
        "protocol_id": SAMPLING_PROTOCOL_ID, "status": "completed",
        "estimator_id": ASR_ESTIMATOR_ID, "samples": samples, "k_values": ks,
        **counts,
        "asr_at_1": endpoints[1]["rate"],
        "asr_at_4": endpoints[min(4, samples)]["rate"] if samples >= 4 else None,
        "asr_at_16": endpoints[samples]["rate"] if samples >= 16 else None,
        "sampled_asr_at_1": endpoints[1]["rate"],
        "sampled_asr_at_1_ci95_low": endpoints[1]["ci95_low"],
        "sampled_asr_at_1_ci95_high": endpoints[1]["ci95_high"],
        "sampled_asr_at_1_missing_lower_bound": endpoints[1]["missing_lower_bound"],
        "sampled_asr_at_1_missing_upper_bound": endpoints[1]["missing_upper_bound"],
        "asr_endpoints": {str(k): value for k, value in endpoints.items()},
        "sampled_not_greedy": True,
        "interpretation": "ASR@k uses k of the same fixed samples; it is not an adaptive attack budget. Greedy ASR is reported separately by the v2 evaluation.",
    }
    return summary


def judge(config: dict[str, Any], run_ids: list[str], samples: int) -> None:
    import torch

    model, tokenizer, scoring = load_harmbench_classifier(config)
    judge_hash = object_hash(scoring)
    for run_id in run_ids:
        directory = sampling_dir(config, run_id)
        manifest = _load_manifest(directory)
        if manifest.get("protocol_id") != SAMPLING_PROTOCOL_ID:
            raise ValueError(f"{run_id}: sampling protocol mismatch")
        if manifest.get("samples") != samples:
            raise ValueError(f"{run_id}: declared sample count differs from --samples")
        if manifest.get("code_sha256") != sampling_code_hashes():
            raise ValueError(f"{run_id}: evaluation code changed since generation")
        expected = _selection(config, None if manifest.get("n") == 200 else int(manifest["n"]))
        if manifest.get("behavior_sha256") != _behavior_identity(expected):
            raise ValueError(f"{run_id}: behavior set or attack wrapper changed")
        rows = _validate_generated(directory, manifest, samples, expected)
        adapter = manifest.get("adapter_identity")
        if adapter is not None:
            if not Path(adapter["path"]).is_dir():
                raise ValueError(f"{run_id}: evaluated adapter is missing")
            if model_identity(adapter["path"], adapter=True)["sha256"] != adapter["sha256"]:
                raise ValueError(f"{run_id}: evaluated adapter changed after generation")
        training = manifest.get("training_identity") or {}
        if training.get("run_json_sha256"):
            run_json = Path(resolve_path(config, config["train"]["output_root"])) / run_id / "run.json"
            if not run_json.exists() or file_hash(run_json) != training["run_json_sha256"]:
                raise ValueError(f"{run_id}: training metadata changed after sampling")
        flat = classifier_inputs(rows)
        labels, raw_labels = classify_harmbench_rows(model, tokenizer, flat)
        per_prompt = [labels[index * samples : (index + 1) * samples] for index in range(len(rows))]
        summary = _judge_summary(per_prompt, samples, int(config["seed"]))
        judged_path = directory / "harmbench16_judged.jsonl"
        write_jsonl(judged_path, [
            {"behavior_id": row["behavior_id"], "behavior": row["behavior"],
             "labels": per_prompt[index],
             "judge_raw": raw_labels[index * samples : (index + 1) * samples]}
            for index, row in enumerate(rows)
        ])
        count_rows = []
        for index, row in enumerate(rows):
            harmful = sum(label is True or label == 1 for label in per_prompt[index])
            unknown = sum(label is None for label in per_prompt[index])
            count_rows.append({"behavior_id": row["behavior_id"], "harmful": harmful,
                               "unknown": unknown, "samples": samples})
        counts_path = directory / "harmbench16_prompt_counts.json"
        write_json(counts_path, {
            "protocol_id": SAMPLING_PROTOCOL_ID, "estimator_id": ASR_ESTIMATOR_ID,
            "samples": samples, "rows": count_rows,
        })
        summary.update(
            judged_sha256=file_hash(judged_path),
            prompt_counts_sha256=file_hash(counts_path),
            scoring=scoring,
            scoring_valid=summary["n_unknown_labels"] == 0,
            valid=True,
            completed_at=now_cst(),
        )
        summary_path = directory / "harmbench16_summary.json"
        write_json(summary_path, summary)
        evaluation_components = {
            "protocol_id": SAMPLING_PROTOCOL_ID, "samples": samples,
            "behavior_source_sha256": manifest["behavior_source_sha256"],
            "behavior_sha256": manifest["behavior_sha256"],
            "attack_prompt_sha256": manifest["attack_prompt_sha256"],
            "decoding": manifest["decoding"], "chat_template_sha256": manifest["chat_template_sha256"],
            "model_identity_sha256": manifest["model_identity"]["sha256"],
            "adapter_identity_sha256": adapter["sha256"] if adapter else None,
            "judge_sha256": judge_hash, "code_sha256": manifest["code_sha256"],
            "raw_sha256": manifest["raw_sha256"], "judged_sha256": summary["judged_sha256"],
            "prompt_counts_sha256": summary["prompt_counts_sha256"],
            "summary_sha256": file_hash(summary_path),
        }
        manifest.update(
            status="completed", valid=True, judge=scoring, judge_sha256=judge_hash,
            judged_sha256=summary["judged_sha256"],
            prompt_counts_sha256=summary["prompt_counts_sha256"],
            summary_sha256=file_hash(summary_path),
            evaluation_components=evaluation_components,
            evaluation_identity=object_hash(evaluation_components),
            training_identity_sha256=object_hash(manifest["training_identity"]),
            completed_at=now_cst(),
        )
        write_json(directory / MANIFEST_NAME, manifest)
        print(
            f"[{run_id}] sampled ASR@1={summary['asr_at_1']} "
            f"ASR@4={summary['asr_at_4']} ASR@16={summary['asr_at_16']} "
            f"unknown={summary['n_unknown_labels']}", flush=True,
        )
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--run", action="append", required=True, help="run_id:adapter_path or run_id:none")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-behaviors", type=int, default=None)
    parser.add_argument("--phase", choices=["generate", "judge", "all"], default="all")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    specs = [spec.split(":", 1) for spec in args.run]
    if any(len(spec) != 2 for spec in specs):
        parser.error("--run must be run_id:adapter_path (or run_id:none)")
    if args.phase in {"generate", "all"}:
        for run_id, adapter in specs:
            generate(config, run_id, adapter, args.samples, args.max_behaviors)
    if args.phase in {"judge", "all"}:
        judge(config, [run_id for run_id, _ in specs], args.samples)


if __name__ == "__main__":
    main()
