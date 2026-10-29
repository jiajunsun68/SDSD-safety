"""Build the fixed SDSD-Safety training/dev/test splits.

Risk prompts come from PKU-SafeRLHF-30K.  We keep prompts for which exactly one
of the two recorded responses is labelled safe, and use that safe response as the
external demonstration.  Normal prompts and responses come from alpaca-cleaned.

The evaluation benchmarks (HarmBench, XSTest, GSM8K) are never used for
training; exact prompt matches against HarmBench/XSTest are removed from the
training pools as a leakage guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from .utils import load_config, now_cst, resolve_path, sha256_file, sha256_text, write_json, write_jsonl

HARMFUL_RE = re.compile(
    r"\b(bomb|explosive|weapon|poison|nerve agent|meth|heroin|cocaine|suicide|self[- ]harm|"
    r"murder|assassinate|terrorist|hack|malware|ransomware|phishing|stalk|harass|exploit|"
    r"child sexual|csam|illegal drug|trafficking)\b",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"^\s*(i'?m sorry|i am sorry|i apologize|i apologise|sorry,|i cannot|i can't|"
    r"i am unable|i'?m unable|i won'?t|i will not|as an ai)",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def load_pku(path: str) -> list[dict[str, Any]]:
    from datasets import Dataset

    dataset = Dataset.from_file(path)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        prompt = str(item.get("prompt", "")).strip()
        safe_flags = [
            str(item.get("is_response_0_safe", "")).lower() == "true",
            str(item.get("is_response_1_safe", "")).lower() == "true",
        ]
        if sum(safe_flags) != 1:
            continue
        safe_index = 0 if safe_flags[0] else 1
        response = str(item.get(f"response_{safe_index}", "")).strip()
        if not prompt or not response:
            continue
        rows.append(
            {
                "id": f"pku-train-{index:06d}",
                "prompt": prompt,
                "kind": "harmful",
                "external_response": response,
                "source": "PKU-SafeRLHF-30K",
            }
        )
    return rows


def load_alpaca(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        instruction = str(item.get("instruction", "")).strip()
        extra_input = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        prompt = instruction if not extra_input else f"{instruction}\n\nInput:\n{extra_input}"
        rows.append(
            {
                "id": f"alpaca-train-{index:06d}",
                "prompt": prompt,
                "kind": "normal",
                "external_response": output,
                "source": "alpaca-cleaned",
            }
        )
    return rows


def load_eval_prompts(config: dict[str, Any]) -> set[str]:
    prompts: set[str] = set()
    with open(resolve_path(config, config["sources"]["xstest_csv"]), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompts.add(normalize(row["prompt"]))
    with open(resolve_path(config, config["sources"]["harmbench_csv"]), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompts.add(normalize(row["Behavior"]))
    return prompts


def valid_harmful(row: dict[str, Any], min_response_chars: int) -> bool:
    prompt = row["prompt"]
    response = row["external_response"]
    return (
        10 <= len(prompt) <= 800
        and min_response_chars <= len(response) <= 4000
        and not REFUSAL_RE.match(prompt)
    )


def valid_normal(row: dict[str, Any], min_response_chars: int) -> bool:
    prompt = row["prompt"]
    response = row["external_response"]
    return (
        10 <= len(prompt) <= 800
        and min_response_chars <= len(response) <= 4000
        and not HARMFUL_RE.search(prompt)
        and not REFUSAL_RE.match(response)
    )


def dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in rows:
        key = normalize(row["prompt"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def take(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    forbidden: set[str],
    split_name: str,
) -> list[dict[str, Any]]:
    rng.shuffle(rows)
    chosen = []
    for row in rows:
        if normalize(row["prompt"]) in forbidden:
            continue
        row = dict(row)
        row["split"] = split_name
        chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"{split_name}: wanted {count} rows, found {len(chosen)}")
    return chosen


def build(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    split_cfg = config["split"]
    min_response_chars = int(config["data"]["min_response_chars"])
    rng = random.Random(seed)

    pku_rows = load_pku(resolve_path(config, config["sources"]["pku_train_arrow"]))
    alpaca_rows = load_alpaca(resolve_path(config, config["sources"]["alpaca_json"]))
    pku_rows = dedup([row for row in pku_rows if valid_harmful(row, min_response_chars)])
    alpaca_rows = dedup([row for row in alpaca_rows if valid_normal(row, min_response_chars)])

    eval_prompts = load_eval_prompts(config)
    pku_rows = [row for row in pku_rows if normalize(row["prompt"]) not in eval_prompts]
    alpaca_rows = [row for row in alpaca_rows if normalize(row["prompt"]) not in eval_prompts]

    harmful_train = take(pku_rows, split_cfg["harmful_train"], rng, set(), "train")
    forbidden = {normalize(row["prompt"]) for row in harmful_train}
    harmful_dev = take(pku_rows, split_cfg["harmful_dev"], rng, forbidden, "dev")
    forbidden |= {normalize(row["prompt"]) for row in harmful_dev}
    harmful_test = take(pku_rows, split_cfg["harmful_test"], rng, forbidden, "test")

    forbidden = {normalize(row["prompt"]) for row in harmful_train + harmful_dev + harmful_test}
    normal_train = take(alpaca_rows, split_cfg["normal_train"], rng, forbidden, "train")
    forbidden |= {normalize(row["prompt"]) for row in normal_train}
    normal_dev = take(alpaca_rows, split_cfg["normal_dev"], rng, forbidden, "dev")
    forbidden |= {normalize(row["prompt"]) for row in normal_dev}
    normal_test = take(alpaca_rows, split_cfg["normal_test"], rng, forbidden, "test")

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
        "filters": {
            "harmful": "exactly one response safe; use the safe response as external demo",
            "normal": "alpaca instruction/output; keyword risk filter; non-refusal output",
            "min_response_chars": min_response_chars,
            "eval_prompt_leakage_guard": ["HarmBench", "XSTest"],
        },
        "counts": {
            "pku_pool": len(pku_rows),
            "alpaca_pool": len(alpaca_rows),
            "train": {"total": len(train), "harmful": len(harmful_train), "normal": len(normal_train)},
            "dev": {"total": len(dev), "harmful": len(harmful_dev), "normal": len(normal_dev)},
            "test": {"total": len(test), "harmful": len(harmful_test), "normal": len(normal_test)},
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "prompt_sha256": sha256_text("\n".join(row["prompt"] for row in rows)),
            }
            for (name, path), rows in zip(paths.items(), (train, dev, test))
        },
    }
    write_json(root / "data" / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    manifest = build(config)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
