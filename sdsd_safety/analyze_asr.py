"""Validate and compare repeated-sampling HarmBench endpoints by request.

This reads only completed, protocol-stamped sampling artifacts. It emits a
risk-versus-k table/figure and paired request-level differences; it never
reclassifies or regenerates model responses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate import file_hash, sampling_dir
from .metrics import ASR_SAMPLING_PROTOCOL_ID, paired_asr_difference
from .utils import load_config, read_jsonl, write_json


def load_run(config: dict[str, Any], run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[bool | None]]]:
    directory = sampling_dir(config, run_id)
    summary_path = directory / "harmbench16_summary.json"
    manifest_path = directory / "sampling_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{run_id}: missing sampling summary/manifest")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (summary.get("protocol_id") != ASR_SAMPLING_PROTOCOL_ID
            or summary.get("status") != "completed" or summary.get("valid") is not True
            or manifest.get("protocol_id") != ASR_SAMPLING_PROTOCOL_ID
            or manifest.get("status") != "completed" or manifest.get("valid") is not True
            or manifest.get("summary_sha256") != file_hash(summary_path)
            or manifest.get("raw_sha256") != file_hash(directory / "harmbench16.jsonl")
            or summary.get("judged_sha256") != file_hash(directory / "harmbench16_judged.jsonl")):
        raise ValueError(f"{run_id}: sampling artifacts are incomplete or changed")
    judged_rows = read_jsonl(directory / "harmbench16_judged.jsonl")
    by_id: dict[str, list[bool | None]] = {}
    ordered = []
    for row in judged_rows:
        if row["behavior_id"] in by_id:
            raise ValueError(f"{run_id}: duplicate behavior {row['behavior_id']}")
        by_id[row["behavior_id"]] = row["labels"]
        ordered.append(row)
    if len(ordered) != 200 or any(len(labels) != 16 for labels in by_id.values()):
        raise ValueError(f"{run_id}: expected 200 behaviors x 16 samples")
    return summary, ordered, by_id


def validate_alignment(runs: dict[str, dict[str, list[bool | None]]]) -> list[str]:
    order = list(next(iter(runs.values())).keys())
    if len(order) != 200 or len(set(order)) != 200:
        raise ValueError("sampling runs do not contain 200 unique behavior IDs")
    for run_id, labels in runs.items():
        if set(labels) != set(order):
            raise ValueError(f"{run_id}: behavior IDs differ from the reference run")
    return order


def event_at_k(labels: list[bool | None], k: int) -> float | None:
    from .metrics import sampling_prompt_statistics

    point = sampling_prompt_statistics([labels], k)["point"][0]
    return point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--runs", required=True, help="comma-separated run IDs")
    parser.add_argument("--reference", default="base_v2")
    parser.add_argument("--bootstrap-seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--output-prefix", default="asr_round1")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    run_ids = [item.strip() for item in args.runs.split(",") if item.strip()]
    summaries, orders, labels_by_run = {}, {}, {}
    for run_id in run_ids:
        summary, ordered, labels = load_run(config, run_id)
        summaries[run_id], orders[run_id], labels_by_run[run_id] = summary, ordered, labels
    order = validate_alignment(labels_by_run)
    root = Path(config["project_root"])
    report: dict[str, Any] = {
        "protocol_id": ASR_SAMPLING_PROTOCOL_ID,
        "runs": run_ids, "reference": args.reference,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "n_requests": len(order),
        "endpoints": {},
        "paired": {},
        "per_request_events": {},
    }
    for run_id in run_ids:
        report["endpoints"][run_id] = summaries[run_id]["asr_endpoints"]
    comparisons = []
    if args.reference not in labels_by_run:
        raise ValueError("reference run is not included")
    for run_id in run_ids:
        if run_id != args.reference:
            comparisons.append((run_id, args.reference))
    for a, b in (("sdsd_v2_anchor_k1_100_seed42", "sdsd_v2_k1_100_seed42"),
                 ("sdsd_v2_external_k1_100_seed42", "sdsd_v2_k1_100_seed42"),
                 ("sdsd_v2_anchor_k1_100_seed42", "sdsd_v2_external_k1_100_seed42")):
        if a in labels_by_run and b in labels_by_run and (a, b) not in comparisons:
            comparisons.append((a, b))
    for a, b in comparisons:
        for k in (1, 4, 16):
            key = f"{a}__minus__{b}__at_{k}"
            report["paired"][key] = paired_asr_difference(
                [labels_by_run[a][behavior_id] for behavior_id in order],
                [labels_by_run[b][behavior_id] for behavior_id in order],
                k, args.bootstrap_seed + k * 1009, args.bootstrap_replicates,
            )
    # Per-request event indicators at k=16 for the main method comparison, plus
    # the union of event sets. This is descriptive, not a significance test.
    pairs = [(a, b) for a, b in comparisons]
    for a, b in pairs:
        rows = []
        for behavior_id in order:
            event_a = event_at_k(labels_by_run[a][behavior_id], 16)
            event_b = event_at_k(labels_by_run[b][behavior_id], 16)
            if event_a is None or event_b is None:
                category = "unknown"
            elif event_a == 1 and event_b == 0:
                category = "only_a"
            elif event_a == 0 and event_b == 1:
                category = "only_b"
            elif event_a == 1:
                category = "both"
            else:
                category = "neither"
            rows.append({"behavior_id": behavior_id, "event_a": event_a, "event_b": event_b,
                         "category": category})
        counts = {category: sum(row["category"] == category for row in rows)
                  for category in ("only_a", "only_b", "both", "neither", "unknown")}
        report["per_request_events"][f"{a}__minus__{b}"] = {"k": 16, "counts": counts, "rows": rows}

    output_dir = root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.output_prefix}.json").write_text(json.dumps(report, indent=2) + "\n")

    table_dir = root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / f"{args.output_prefix}_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run_a", "run_b", "k", "difference", "ci95_low", "ci95_high",
                         "difference_lower_bound", "difference_upper_bound",
                         "n_paired_complete_requests", "n_requests_with_unknown", "ci_excludes_zero"])
        for key, value in report["paired"].items():
            run_a, rest = key.split("__minus__", 1)
            run_b, k = rest.rsplit("__at_", 1)
            writer.writerow([run_a, run_b, k, value["difference"], value["ci95_low"], value["ci95_high"],
                             value["difference_lower_bound"], value["difference_upper_bound"],
                             value["n_paired_complete_requests"], value["n_requests_with_unknown"],
                             value["ci_excludes_zero"]])
    lines = [
        f"# Repeated-sampling paired analysis ({args.output_prefix})",
        "",
        f"Protocol `{ASR_SAMPLING_PROTOCOL_ID}`; {len(order)} fixed official behaviors; "
        "request-level paired bootstrap. Negative differences mean run A has lower ASR.",
        "",
        "| A minus B | k | ΔASR | 95% bootstrap CI | conservative missing bound | paired requests | CI excludes 0 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, value in report["paired"].items():
        run_a, rest = key.split("__minus__", 1)
        run_b, k = rest.rsplit("__at_", 1)
        fmt = lambda x: "—" if x is None else f"{100*x:+.2f}"
        lines.append(
            f"| {run_a} − {run_b} | {k} | {fmt(value['difference'])} | "
            f"[{fmt(value['ci95_low'])}, {fmt(value['ci95_high'])}] | "
            f"[{fmt(value['difference_lower_bound'])}, {fmt(value['difference_upper_bound'])}] | "
            f"{value['n_paired_complete_requests']} | {value['ci_excludes_zero']} |"
        )
    lines += ["", "## ASR endpoints", "", "| Run | ASR@1 | ASR@4 | ASR@16 |", "|---|---:|---:|---:|"]
    for run_id in run_ids:
        endpoints = report["endpoints"][run_id]
        lines.append("| " + run_id + " | " + " | ".join(
            "—" if endpoints[str(k)]["rate"] is None else f"{100*endpoints[str(k)]['rate']:.2f}"
            for k in (1, 4, 16)) + " |")
    lines += ["", "## Per-request ASR@16 event overlap", "",
              "| A minus B | only A harmful | only B harmful | both | neither | unknown |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, value in report["per_request_events"].items():
        counts = value["counts"]
        lines.append(f"| {key.replace('__minus__', ' − ')} | {counts['only_a']} | {counts['only_b']} | "
                     f"{counts['both']} | {counts['neither']} | {counts['unknown']} |")
    (table_dir / f"{args.output_prefix}_paired.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Risk-versus-k figure. Matplotlib is optional; the CSV/Markdown remain canonical.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_dir = root / "results" / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for run_id in run_ids:
            endpoints = report["endpoints"][run_id]
            xs = [1, 4, 16]
            ys = [endpoints[str(k)]["rate"] for k in xs]
            lows = [endpoints[str(k)]["ci95_low"] for k in xs]
            highs = [endpoints[str(k)]["ci95_high"] for k in xs]
            if any(value is None for value in ys):
                continue
            ax.plot(xs, [100 * value for value in ys], marker="o", label=run_id)
            if all(value is not None for value in lows + highs):
                ax.fill_between(xs, [100 * value for value in lows], [100 * value for value in highs], alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], ["1", "4", "16"])
        ax.set_xlabel("k (of the same 16 samples)")
        ax.set_ylabel("HarmBench ASR@k (%)")
        ax.set_title("Fixed-prompt direct-request repeated sampling")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{args.output_prefix}_asr_vs_k.png", dpi=200)
        plt.close(fig)
    except Exception as error:
        print(f"figure skipped: {type(error).__name__}: {error}")
    print(f"wrote {output_dir / (args.output_prefix + '.json')} and paired tables")


if __name__ == "__main__":
    main()
