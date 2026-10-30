"""Fixed-attack repeated-sampling evaluation for HarmBench.

Direct-request sampling saturated at the floor in Round 1, so this module adds
one *fixed* attack protocol for development: the official HarmBench
HumanJailbreaks prompt at a frozen index (source file hash + index recorded),
prepended to each HarmBench standard-validation behavior exactly as the official
baseline does.  All target runs use the same attack prompts, decoding budget and
HarmBench classifier.  Attack prompts and original behaviors remain separate
fields; the classifier sees the original behavior only.

This is a fixed-prompt protocol, not a transfer/adaptive attack.  It is kept in
its own protocol id/directory so the frozen direct-request artifacts
(asr-sample-v3) are never overwritten.  No training and no model export.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import random
import shlex
import statistics
import sys
import time
from typing import Any

from .evaluate import (
    classify_harmbench_rows,
    file_hash,
    git_commit,
    git_worktree_status,
    load_harmbench_classifier,
    load_model_for_eval,
    model_identity,
    training_provenance,
)
from .metrics import asr_at_k_table
from .modeling import generate_rollouts
from .utils import load_config, now_cst, read_jsonl, resolve_path, set_seed, write_json, write_jsonl

PROTOCOL_ID = "sdsd-safety-hb-humanjb-fixed5-dev-v1"
ATTACK_DIR = Path("results") / "raw_attack" / "humanjb-fixed5-v1"
TABLE_NAME = "attack_humanjb_fixed5.md"
REPORT_NAME = "attack_humanjb_fixed5.json"
JAILBREAK_SOURCE = Path("/data4/sjj/alignment/harmbench_official_8e1604d/baselines/human_jailbreaks/jailbreaks.py")
JAILBREAK_INDEX = 5
BEHAVIORS_CSV = Path("/data4/sjj/alignment/harmbench_official_8e1604d/data/behavior_datasets/harmbench_behaviors_text_val.csv")
EXPECTED_BEHAVIORS = 41
MAX_NEW_TOKENS = 320
CODE_FILES = ("harmbench_attack.py", "metrics.py", "evaluate.py", "modeling.py", "judging.py", "utils.py")
DEFAULT_RUNS = {
    "base_v2": None,
    "opsa_v2_100_seed42": "runs/opsa_v2_100_seed42/checkpoint",
    "sdsd_v2_k1_100_seed42": "runs/sdsd_v2_k1_100_seed42/checkpoint",
    "sdsd_v2_anchor_k1_100_seed42": "runs/sdsd_v2_anchor_k1_100_seed42/checkpoint",
    "sdsd_v2_external_k1_100_seed42": "runs/sdsd_v2_external_k1_100_seed42/checkpoint",
}


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).parent
    return {name: file_hash(package / name) for name in CODE_FILES}


def load_jailbreak() -> tuple[str, dict[str, Any]]:
    source = JAILBREAK_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    values = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "JAILBREAKS"
                                                for target in node.targets):
            values = ast.literal_eval(node.value)
            break
    if not isinstance(values, list) or not 0 <= JAILBREAK_INDEX < len(values):
        raise ValueError("official JAILBREAKS list is missing or the frozen index is out of range")
    prompt = values[JAILBREAK_INDEX]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("frozen jailbreak prompt is empty")
    identity = {
        "source_path": str(JAILBREAK_SOURCE), "source_sha256": file_hash(JAILBREAK_SOURCE),
        "index": JAILBREAK_INDEX, "n_prompts": len(values),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt),
    }
    return prompt, identity


def load_standard_val_behaviors() -> list[dict[str, Any]]:
    with BEHAVIORS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"BehaviorID", "Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"validation source lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for row in reader:
            if row["FunctionalCategory"] != "standard":
                continue
            if row["ContextString"].strip() or {"context", "hash_check"}.intersection(
                    tag.strip() for tag in row["Tags"].split(",")):
                raise ValueError(f"standard validation behavior needs contextual/hash evaluation: {row['BehaviorID']}")
            rows.append({"id": row["BehaviorID"], "prompt": row["Behavior"],
                         "category": row["SemanticCategory"]})
    if len(rows) != EXPECTED_BEHAVIORS or len({row["id"] for row in rows}) != EXPECTED_BEHAVIORS:
        raise ValueError(f"expected {EXPECTED_BEHAVIORS} unique standard behaviors, got {len(rows)}")
    return rows


def attack_prompt(jailbreak: str, behavior: str) -> str:
    # Exactly the official HumanJailbreaks construction: jailbreak text, blank line, behavior.
    return f"{jailbreak}\n\n{behavior}"


def run_directory(config: dict[str, Any], run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    directory = Path(config["project_root"]) / ATTACK_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _adapter_identity(config: dict[str, Any], adapter: str | None) -> dict[str, Any] | None:
    if not adapter:
        return None
    path = str(resolve_path(config, adapter))
    return {**model_identity(path, adapter=True), "path": path}


def _prompt_hash(behaviors: list[dict[str, Any]], jailbreak: str) -> str:
    payload = [{"behavior_id": row["id"], "behavior": row["prompt"],
                "attack_prompt": attack_prompt(jailbreak, row["prompt"])} for row in behaviors]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def generate(config: dict[str, Any], run_id: str, adapter: str | None, samples: int) -> Path:
    directory = run_directory(config, run_id)
    manifest_path = directory / "attack_manifest.json"
    raw_path = directory / "harmbench_attack.jsonl"
    if raw_path.exists() or manifest_path.exists():
        raise FileExistsError(f"attack output already exists (no overwrite): {directory}")

    import torch

    jailbreak, jailbreak_identity = load_jailbreak()
    behaviors = load_standard_val_behaviors()
    prompts = [attack_prompt(jailbreak, row["prompt"]) for row in behaviors]
    decoding = {
        "do_sample": True, "samples": samples, "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "batch_size": 4,
        "max_prompt_tokens": int(config["data"]["max_prompt_tokens"]), "seed": int(config["seed"]),
    }
    set_seed(int(config["seed"]))
    model, tokenizer = load_model_for_eval(config, adapter)
    started = time.perf_counter()
    results = generate_rollouts(
        model, tokenizer, prompts, num_rollouts=samples,
        max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, top_p=1.0,
        batch_size=4, greedy=False, system=None,
        max_prompt_tokens=int(config["data"]["max_prompt_tokens"]),
    )
    elapsed = time.perf_counter() - started
    rows = []
    for behavior, samples_out in zip(behaviors, results):
        texts = [sample["text"] for sample in samples_out]
        lengths = [len(sample["token_ids"]) for sample in samples_out]
        rows.append({
            "behavior_id": behavior["id"], "behavior": behavior["prompt"],
            "attack_prompt": attack_prompt(jailbreak, behavior["prompt"]),
            f"attack_mode": f"humanjb-idx{JAILBREAK_INDEX}", "category": behavior["category"],
            "responses": texts, "response_token_counts": lengths,
            "length_hit": [int(length >= MAX_NEW_TOKENS) for length in lengths],
        })
    write_jsonl(raw_path, rows)
    identity = {
        "protocol_id": PROTOCOL_ID, "status": "generated", "valid": False,
        "run_id": run_id, "samples": samples, "n": len(rows), "seed": int(config["seed"]),
        "started_at": now_cst(), "seconds": elapsed,
        "git_commit": git_commit(), **git_worktree_status(),
        "command": shlex.join([sys.executable, *sys.argv]), "argv": [sys.executable, *sys.argv],
        "code_sha256": _code_hashes(), "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config": config, "decoding": decoding,
        "model_identity": model_identity(resolve_path(config, config["model"]["path"])),
        "adapter_identity": _adapter_identity(config, adapter),
        "training_identity": training_provenance(config, run_id, adapter),
        "behavior_source_path": str(BEHAVIORS_CSV), "behavior_source_sha256": file_hash(BEHAVIORS_CSV),
        "behavior_ids": [row["id"] for row in behaviors],
        "behavior_prompt_sha256": hashlib.sha256(
            json.dumps([row["prompt"] for row in behaviors], ensure_ascii=False).encode()).hexdigest(),
        "attack_prompt_sha256": _prompt_hash(behaviors, jailbreak),
        "jailbreak": jailbreak_identity,
        "raw_sha256": file_hash(raw_path),
        "length_hit_rate": sum(sum(row["length_hit"]) for row in rows) / (len(rows) * samples),
    }
    write_json(manifest_path, identity)
    print(f"[{run_id}] generated {len(rows)} x {samples} attack samples in {elapsed:.1f}s; "
          f"length_hit_rate={identity['length_hit_rate']:.3f}", flush=True)
    return directory


def judge(config: dict[str, Any], run_id: str, samples: int, bootstrap_seed: int,
          bootstrap_replicates: int) -> dict[str, Any]:
    directory = run_directory(config, run_id)
    manifest_path = directory / "attack_manifest.json"
    raw_path = directory / "harmbench_attack.jsonl"
    if not manifest_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError(f"missing attack generation for {run_id}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("samples") != samples:
        raise ValueError(f"{run_id}: protocol/sample mismatch")
    if manifest.get("raw_sha256") != file_hash(raw_path):
        raise ValueError(f"{run_id}: raw attack output changed")
    rows = read_jsonl(raw_path)
    if len(rows) != EXPECTED_BEHAVIORS or any(len(row["responses"]) != samples for row in rows):
        raise ValueError(f"{run_id}: expected {EXPECTED_BEHAVIORS} behaviors x {samples} responses")

    import torch

    model, tokenizer, _ = load_harmbench_classifier(config)
    flat = [{"behavior": row["behavior"], "response": response}
            for row in rows for response in row["responses"]]
    labels, raw_labels = classify_harmbench_rows(model, tokenizer, flat)
    per_prompt: list[list[bool | None]] = []
    cursor = 0
    for row in rows:
        per_prompt.append(labels[cursor:cursor + samples])
        cursor += samples
    ks = sorted({1, min(4, samples), samples})
    endpoints = asr_at_k_table(per_prompt, ks, bootstrap_seed, bootstrap_replicates)
    judged = []
    cursor = 0
    for row in rows:
        judged.append({
            "behavior_id": row["behavior_id"], "behavior": row["behavior"],
            "attack_prompt": row["attack_prompt"], "attack_mode": row["attack_mode"],
            "responses": row["responses"], "labels": labels[cursor:cursor + samples],
            "raw_labels": raw_labels[cursor:cursor + samples],
        })
        cursor += samples
    judged_path = directory / "harmbench_attack_judged.jsonl"
    write_jsonl(judged_path, judged)
    summary = {
        "protocol_id": PROTOCOL_ID, "status": "completed", "valid": True,
        "run_id": run_id, "n": len(rows), "samples": samples,
        "n_unknown_labels": sum(label is None for label in labels),
        "asr_endpoints": {str(k): value for k, value in endpoints.items()},
        "judged_sha256": file_hash(judged_path),
        "judge_model": manifest.get("config", {}).get("sources", {}).get("harmbench_classifier"),
        "completed_at": now_cst(),
    }
    summary["valid"] = summary["n_unknown_labels"] == 0
    write_json(directory / "harmbench_attack_summary.json", summary)
    manifest["status"] = "completed"
    manifest["valid"] = summary["valid"]
    manifest["judged_sha256"] = summary["judged_sha256"]
    manifest["summary_sha256"] = file_hash(directory / "harmbench_attack_summary.json")
    write_json(manifest_path, manifest)
    rate_4 = endpoints.get(4, {}).get("rate")
    rate_16 = endpoints.get(16, {}).get("rate")
    print(f"[{run_id}] attack ASR@1={endpoints[1]['rate']:.4f} "
          f"ASR@4={rate_4 if rate_4 is not None else '—'} "
          f"ASR@16={rate_16 if rate_16 is not None else '—'} "
          f"unknown={summary['n_unknown_labels']}", flush=True)
    return summary


def analyze(config: dict[str, Any], run_ids: list[str]) -> dict[str, Any]:
    root = Path(config["project_root"])
    summaries = {}
    labels_by_run = {}
    for run_id in run_ids:
        directory = root / ATTACK_DIR / run_id
        summary = json.loads((directory / "harmbench_attack_summary.json").read_text())
        if summary.get("protocol_id") != PROTOCOL_ID or summary.get("valid") is not True:
            raise ValueError(f"{run_id}: attack summary invalid")
        rows = read_jsonl(directory / "harmbench_attack_judged.jsonl")
        labels_by_run[run_id] = {row["behavior_id"]: row["labels"] for row in rows}
        summaries[run_id] = summary
    order = list(next(iter(labels_by_run.values())).keys())
    for run_id, labels in labels_by_run.items():
        if set(labels) != set(order):
            raise ValueError(f"{run_id}: behavior IDs differ")
    from .metrics import paired_asr_difference
    reference = run_ids[0]
    paired = {}
    for run_id in run_ids[1:]:
        for k in (1, 4, 16):
            paired[f"{run_id}__minus__{reference}__at_{k}"] = paired_asr_difference(
                [labels_by_run[run_id][bid] for bid in order],
                [labels_by_run[reference][bid] for bid in order],
                k, int(config["acl"]["bootstrap_seed"]) + k * 1009,
                int(config["acl"]["bootstrap_replicates"]),
            )
    report = {"protocol_id": PROTOCOL_ID, "runs": run_ids, "reference": reference,
              "n_behaviors": len(order), "endpoints": {r: summaries[r]["asr_endpoints"] for r in run_ids},
              "paired": paired, "completed_at": now_cst()}
    analysis_dir = root / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir / REPORT_NAME, report)
    table_dir = root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Fixed-attack repeated-sampling ASR", "",
             f"Protocol `{PROTOCOL_ID}`; HarmBench standard behaviors (n={len(order)}; source/hash in manifest); "
             f"official HumanJailbreaks prompt index {JAILBREAK_INDEX} (source hash recorded); "
             f"temperature 1, top-p 1, top-k 0, max {MAX_NEW_TOKENS} new tokens; classifier reads the "
             "original behavior only. This is a fixed prompt protocol, not a transfer or adaptive attack.", "",
             "| Run | ASR@1 % | ASR@4 % | ASR@16 % | Unknown |", "|---|---:|---:|---:|---:|"]
    for run_id in run_ids:
        endpoints = summaries[run_id]["asr_endpoints"]
        fmt = lambda k: "—" if endpoints[str(k)]["rate"] is None else f"{100*endpoints[str(k)]['rate']:.2f}"
        lines.append(f"| {run_id} | {fmt(1)} | {fmt(4)} | {fmt(16)} | {summaries[run_id]['n_unknown_labels']} |")
    lines += ["", "| A minus B | k | ΔASR (pp) | 95% CI | CI excludes 0 |", "|---|---:|---:|---|---|"]
    for key, value in paired.items():
        a, rest = key.split("__minus__", 1)
        b, k = rest.rsplit("__at_", 1)
        fmt = lambda x: "—" if x is None else f"{100*x:+.2f}"
        lines.append(f"| {a} − {b} | {k} | {fmt(value['difference'])} | "
                     f"[{fmt(value['ci95_low'])}, {fmt(value['ci95_high'])}] | {value['ci_excludes_zero']} |")
    (table_dir / TABLE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--phase", choices=("generate", "judge", "all", "analyze"), default="all")
    parser.add_argument("--bootstrap-seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    run_ids = [item.strip() for item in args.runs.split(",") if item.strip()]
    unknown = [run_id for run_id in run_ids if run_id not in DEFAULT_RUNS]
    if unknown:
        raise SystemExit(f"unknown run ids: {unknown}")
    if args.phase in {"generate", "all"}:
        for run_id in run_ids:
            generate(config, run_id, DEFAULT_RUNS[run_id], args.samples)
    if args.phase in {"judge", "all"}:
        for run_id in run_ids:
            judge(config, run_id, args.samples, args.bootstrap_seed, args.bootstrap_replicates)
    if args.phase in {"analyze", "all"}:
        analyze(config, run_ids)


if __name__ == "__main__":
    main()
