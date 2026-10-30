"""Audit request intent and response suitability before allocating fixed splits.

Response labels select candidates, never request intent labels. Sensitive but
benign requests remain eligible. Refusal targets are verified in the next stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
from typing import Any

from .prepare_data import (
    HARMFUL_RE, build_splits, dedup, load_alpaca, load_eval_prompts, load_pku,
    normalize, valid_harmful, valid_normal,
)
from .utils import load_config, now_cst, resolve_path, sha256_file, write_json, write_jsonl

PROTOCOL = "prompt-intent-v2"


def qualify_candidate(row: dict[str, Any], prompt_judgment: dict[str, Any],
                      response_judgment: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Convert audited labels without inferring request intent from an answer."""
    intent = prompt_judgment.get("safety")
    if intent not in {"Safe", "Unsafe"}:
        return None, "request_ambiguous_or_unparsed"
    if response_judgment.get("safety") != "Safe":
        return None, "external_response_not_verified_safe"
    kind = "normal" if intent == "Safe" else "harmful"
    refusal = response_judgment.get("refusal")
    if kind == "normal" and refusal is not False:
        return None, "normal_response_refusing_or_unparsed"
    item = dict(row)
    item.update(kind=kind, prompt_safety=intent, external_safety="Safe",
                external_refusal=refusal, prompt_audit_protocol=PROTOCOL,
                boundary_normal=kind == "normal" and bool(HARMFUL_RE.search(row["prompt"])))
    return item, "accepted"


def candidate_pool(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    minimum = int(config["data"]["min_response_chars"])
    pku = dedup([r for r in load_pku(resolve_path(config, config["sources"]["pku_train_arrow"]))
                 if valid_harmful(r, minimum)])
    alpaca = dedup([r for r in load_alpaca(resolve_path(config, config["sources"]["alpaca_json"]))
                   if valid_normal(r, minimum)])
    forbidden = load_eval_prompts(config)
    pku = [r for r in pku if normalize(r["prompt"]) not in forbidden]
    alpaca = [r for r in alpaca if normalize(r["prompt"]) not in forbidden]
    rng = random.Random(int(config["seed"]) + 7)
    boundary = [r for r in alpaca if HARMFUL_RE.search(r["prompt"])]
    ordinary = [r for r in alpaca if not HARMFUL_RE.search(r["prompt"])]
    rng.shuffle(boundary)
    rng.shuffle(ordinary)
    required = sum(int(config["split"][f"normal_{s}"]) for s in ("train", "dev", "test"))
    budget = max(required, round(required * float(config["data"].get("normal_candidate_multiplier", 2.0))))
    chosen_boundary = boundary[: min(len(boundary), budget // 2)]
    selected = chosen_boundary + ordinary[: budget - len(chosen_boundary)]
    rows = dedup(pku + selected)
    return rows, {"pku_candidates": len(pku), "alpaca_candidates": len(selected),
                  "alpaca_sensitive_candidates": len(chosen_boundary), "total": len(rows),
                  "exact_benchmark_exclusions": ["HarmBench", "XSTest", "GSM8K"]}


def run_audit(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    from .judging import Qwen3GuardJudge

    root = Path(config["project_root"])
    manifest_path = Path(resolve_path(config, config["data"].get("manifest", "data/manifest.json")))
    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL, "created_at": now_cst(), "seed": int(config["seed"]),
        "prompt_audit": {"protocol": PROTOCOL, "completed": False},
        "targets": {"completed": False}, "status": "running",
    }
    write_json(manifest_path, manifest)
    try:
        candidates, counts = candidate_pool(config)
        judge = Qwen3GuardJudge(config["judge"]["path"], batch_size=batch_size,
                                max_new_tokens=min(32, max(16, int(config["judge"]["max_new_tokens"]))))
        kept: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        reasons: Counter = Counter()
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            prompts = [r["prompt"] for r in chunk]
            pjudgments = judge.judge_prompts(prompts)
            rjudgments = judge.judge_responses(prompts, [r["external_response"] for r in chunk])
            for row, pj, rj in zip(chunk, pjudgments, rjudgments, strict=True):
                qualified, reason = qualify_candidate(row, pj, rj)
                reasons[reason] += 1
                if qualified is not None:
                    kept.append(qualified)
                audit_rows.append({"id": row["id"], "source": row["source"],
                                   "request_safety": pj.get("safety"),
                                   "response_safety": rj.get("safety"),
                                   "response_refusal": rj.get("refusal"), "decision": reason})
            print(f"audited {min(start + batch_size, len(candidates))}/{len(candidates)} requests and responses", flush=True)
        write_jsonl(root / "data" / "external_audit.jsonl", audit_rows)
        harmful = [r for r in kept if r["kind"] == "harmful"]
        normal = [r for r in kept if r["kind"] == "normal"]
        splits = build_splits(config, harmful, normal)
        minimum_boundary = int(config["split"].get("min_boundary_normal_per_split", 1))
        for name, rows in splits.items():
            found = sum(bool(r.get("boundary_normal")) for r in rows)
            if found < min(minimum_boundary, int(config["split"][f"normal_{name}"])):
                raise RuntimeError(f"{name}: only {found} verified sensitive-but-benign examples; "
                                   "expand the local candidate pool before training")
        files, split_counts = {}, {}
        for name, rows in splits.items():
            path = Path(resolve_path(config, config["data"][name]))
            write_jsonl(path, rows)
            files[name] = {"path": str(path), "sha256": sha256_file(path)}
            split_counts[name] = {"total": len(rows),
                                  "harmful": sum(r["kind"] == "harmful" for r in rows),
                                  "normal": sum(r["kind"] == "normal" for r in rows),
                                  "boundary_normal": sum(bool(r.get("boundary_normal")) for r in rows)}
        source_hashes = {
            key: {"path": resolve_path(config, config["sources"][key]),
                  "sha256": sha256_file(resolve_path(config, config["sources"][key]))}
            for key in ("pku_train_arrow", "alpaca_json", "xstest_csv", "harmbench_csv", "gsm8k_test_parquet")
        }
        manifest.update(status="audited_pending_targets", files=files, counts=split_counts,
                        sources=source_hashes, candidates=counts, exclusion_counts=dict(reasons),
                        prompt_audit={"protocol": PROTOCOL, "completed": True,
                                      "judge_path": config["judge"]["path"],
                                      "harmful_accepted": len(harmful), "normal_accepted": len(normal),
                                      "note": "Automated intent labels; ambiguous/unparsed requests excluded."})
        write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest["prompt_audit"]["completed"] = False
        write_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    config = load_config(args.config, args.override)
    manifest = run_audit(config, args.batch_size)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
