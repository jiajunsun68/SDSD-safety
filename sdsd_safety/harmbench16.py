"""Fixed-prompt repeated sampling on the official HarmBench standard-200 pool.

ASR@k describes k independent draws at a declared decoding configuration. It is
not an adaptive attack-budget metric. Missing judgments remain unknown; prompt
clusters, rather than flattened responses, are the resampling unit.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from .evaluate import (
    classify_harmbench_rows, file_hash, finish_generation_metadata,
    generation_provenance, harmbench_rows, load_harmbench_classifier, load_model_for_eval,
    raw_dir, select_saved_rows, stamp_summary, start_generation_metadata,
)
from .metrics import binary_summary, wilson_interval
from .utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def repeated_sampling_summary(per_prompt: list[list[bool | int | None]], samples: int, seed: int) -> dict[str, Any]:
    """Point estimates and prompt-level uncertainty, with explicit missingness."""
    import numpy as np

    if samples < 1 or not per_prompt or any(len(row) != samples for row in per_prompt):
        raise ValueError("each nonempty prompt group must contain exactly k judgments")
    groups = [binary_summary(row) for row in per_prompt]
    complete_rates = [group["rate"] for group in groups if group["n_parse_failure"] == 0]
    flat = binary_summary([label for row in per_prompt for label in row])
    events = [True if any(label == 1 for label in row) else None if any(label is None for label in row) else False
              for row in per_prompt]
    at_k = binary_summary(events)
    if complete_rates:
        rates = np.asarray(complete_rates, dtype=float)
        asr1 = float(rates.mean())
        bootstrap = np.random.default_rng(seed).choice(rates, size=(10000, len(rates)), replace=True).mean(axis=1)
        low, high = (float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
        method = "prompt_cluster_percentile_bootstrap"
        if asr1 in (0.0, 1.0):
            # Resampling all-zero/all-one rows is degenerate. The prompt-level
            # Wilson envelope is conservative for a per-response boundary rate;
            # it does not count k draws as k independent prompts.
            low, high = wilson_interval(int(asr1 * len(rates)), len(rates))
            method = "wilson_prompt_level_boundary_envelope"
    else:
        asr1 = low = high = None
        method = "unavailable_no_completely_judged_prompts"
    return {
        "benchmark": f"HarmBench-standard200@{samples}", "n_behaviors": len(per_prompt),
        "n_generations": flat["n_total"], "n_valid_generations": flat["n_valid"],
        "n_parse_failure": flat["n_parse_failure"], "parse_failure_rate": flat["parse_failure_rate"],
        "n_completely_judged_prompts": len(complete_rates),
        "n_prompts_without_any_valid_label": sum(group["n_valid"] == 0 for group in groups),
        "asr_at_1": asr1 if flat["n_parse_failure"] == 0 else None,
        "asr_at_1_ci95_low": low if flat["n_parse_failure"] == 0 else None,
        "asr_at_1_ci95_high": high if flat["n_parse_failure"] == 0 else None,
        "asr_at_1_complete_case_diagnostic": asr1,
        "asr_at_1_ci_method": method,
        "asr_at_1_missing_lower_bound": flat["missing_lower_bound"],
        "asr_at_1_missing_upper_bound": flat["missing_upper_bound"],
        "asr_at_k": at_k["rate"] if at_k["n_parse_failure"] == 0 else None,
        "asr_at_k_ci95_low": at_k["ci95_low"] if at_k["n_parse_failure"] == 0 else None,
        "asr_at_k_ci95_high": at_k["ci95_high"] if at_k["n_parse_failure"] == 0 else None,
        "asr_at_k_counts": at_k,
        "k": samples, "bootstrap_seed": seed,
        "estimand": "fixed-prompt independent draws; uncertainty across prompt clusters; incomplete prompts excluded from the ASR@1 point estimate and bounded separately",
    }


def generate(config: dict[str, Any], run_id: str, adapter: str | None, samples: int, max_behaviors: int | None) -> None:
    if samples < 1 or (max_behaviors is not None and max_behaviors < 1):
        raise ValueError("samples and max_behaviors must be positive")
    rows = harmbench_rows(config)
    if max_behaviors is not None:
        random.Random(int(config["seed"])).shuffle(rows)
        rows = rows[:max_behaviors]
    import torch
    from .modeling import generate_rollouts

    set_seed(int(config["seed"]))
    model, tokenizer = load_model_for_eval(config, adapter)
    decoding = {"do_sample": True, "num_return_sequences": samples,
                "max_new_tokens": int(config["eval"]["max_new_tokens"]),
                "temperature": float(config["eval"].get("sampling_temperature", 1.0)),
                "top_p": float(config["eval"].get("sampling_top_p", 1.0)), "top_k": 0,
                "batch_size": 4, "max_prompt_tokens": int(config["data"]["max_prompt_tokens"])}
    benchmark = f"harmbench{samples}"
    meta = start_generation_metadata(config, run_id, adapter, benchmark, rows, tokenizer, model, decoding)
    rollouts = generate_rollouts(model, tokenizer, [row["prompt"] for row in rows], num_rollouts=samples,
                                 max_new_tokens=decoding["max_new_tokens"], temperature=decoding["temperature"],
                                 top_p=decoding["top_p"], batch_size=4, greedy=False,
                                 max_prompt_tokens=decoding["max_prompt_tokens"])
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    output = []
    for row, group in zip(rows, rollouts, strict=True):
        if len(group) != samples:
            raise RuntimeError(f"expected {samples} samples for {row['id']}, got {len(group)}")
        output.append({**row, "responses": [sample["text"] for sample in group],
                       "response_token_counts": [len(sample["token_ids"]) for sample in group],
                       "finish_reasons": ["eos" if sample["token_ids"] and sample["token_ids"][-1] in eos_ids
                                          else "length" if len(sample["token_ids"]) >= decoding["max_new_tokens"] else "other"
                                          for sample in group]})
    write_jsonl(raw_dir(config, run_id) / f"{benchmark}.jsonl", output)
    finish_generation_metadata(config, run_id, benchmark, meta)
    print(f"[{run_id}] generated {len(output)} x {samples}, seed={config['seed']}", flush=True)
    del model
    torch.cuda.empty_cache()


def judge(config: dict[str, Any], run_ids: list[str], samples: int) -> None:
    expected = {row["id"]: row for row in harmbench_rows(config)}
    selected = {}
    for run_id in run_ids:
        saved = read_jsonl(raw_dir(config, run_id) / f"harmbench{samples}.jsonl")
        subset = [expected[row["id"]] for row in saved if row["id"] in expected]
        if not subset:
            raise ValueError(f"no official standard behaviors in {run_id}")
        rows = select_saved_rows(saved, subset)
        if any(len(row["responses"]) != samples for row in rows):
            raise ValueError(f"{run_id}: requested k={samples} does not match saved response counts")
        selected[run_id] = rows
    import torch

    model, tokenizer, scoring = load_harmbench_classifier(config)
    for run_id, rows in selected.items():
        benchmark = f"harmbench{samples}"
        provenance = generation_provenance(config, run_id, benchmark, rows)
        flat = [{"prompt": row["prompt"], "response": response} for row in rows for response in row["responses"]]
        labels, raw_labels = classify_harmbench_rows(model, tokenizer, flat)
        per_row = [labels[index * samples : (index + 1) * samples] for index in range(len(rows))]
        summary = repeated_sampling_summary(per_row, samples, int(config["seed"]))
        judged_path = raw_dir(config, run_id) / f"{benchmark}_judged.jsonl"
        write_jsonl(judged_path, [{"id": row["id"], "labels": per_row[index],
                                  "judge_raw": raw_labels[index * samples : (index + 1) * samples]}
                                 for index, row in enumerate(rows)])
        summary["judged_sha256"] = file_hash(judged_path)
        stamp_summary(summary, provenance, scoring_valid=summary["n_parse_failure"] == 0,
                      scoring={**scoring, "metric": "fixed_prompt_independent_sampling", "k": samples})
        write_json(raw_dir(config, run_id) / f"{benchmark}_summary.json", summary)
        print(f"[{run_id}] ASR@1={summary['asr_at_1']} ASR@{samples}={summary['asr_at_k']} parse_failures={summary['n_parse_failure']}", flush=True)
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
