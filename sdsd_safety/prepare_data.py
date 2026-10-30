"""Shared candidate loading and audited, disjoint training/dev/test splits.

Candidates come from PKU-SafeRLHF-30K and alpaca-cleaned. For PKU, exactly one
recorded response must be labelled safe to be proposed as an external target.
Both sources require independent request-intent and response auditing.

Response safety labels select candidate answers, never request intent labels.
Run this module (or audit_external) to classify requests before making splits.
Sensitive keywords mark boundary coverage; they do not reject normal requests.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from .utils import resolve_path

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
                "kind": "unverified",
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
                "kind": "unverified",
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
    # Compare raw questions, not the evaluation-only answer-format suffix.
    gsm_path = config["sources"].get("gsm8k_test_parquet")
    if gsm_path:
        import pandas as pd

        frame = pd.read_parquet(resolve_path(config, gsm_path), columns=["question"])
        prompts.update(normalize(str(question)) for question in frame["question"])
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
    if count < 0:
        raise ValueError("split size must be nonnegative")
    if count == 0:
        return []
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


def build_splits(config: dict[str, Any], harmful: list[dict[str, Any]], normal: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Allocate each verified prompt once; retain sensitive-but-benign examples.

    Boundary coverage is descriptive. A keyword does not establish harmlessness;
    every input to this function must already have a prompt-level judgment.
    """
    rng = random.Random(int(config["seed"]))
    share = float(config["split"].get("normal_boundary_fraction", 0.25))
    if not 0.0 <= share <= 1.0:
        raise ValueError("normal_boundary_fraction must be between 0 and 1")
    for kind, rows in (("harmful", harmful), ("normal", normal)):
        expected = "Unsafe" if kind == "harmful" else "Safe"
        if any(row.get("kind") != kind or row.get("prompt_safety") != expected for row in rows):
            raise ValueError("build_splits requires audited request-intent labels")
    pools = {"harmful": dedup(harmful), "normal": dedup(normal)}
    forbidden: set[str] = set()
    splits = {}
    names = ("train", "dev", "test")
    minimum = int(config["split"].get("min_boundary_normal_per_split", 1))
    if minimum < 0:
        raise ValueError("min_boundary_normal_per_split cannot be negative")
    for index, name in enumerate(names):
        bad = take(pools["harmful"], int(config["split"][f"harmful_{name}"]), rng, forbidden, name)
        forbidden.update(normalize(row["prompt"]) for row in bad)
        count = int(config["split"][f"normal_{name}"])
        boundary = [row for row in pools["normal"] if row.get("boundary_normal") and normalize(row["prompt"]) not in forbidden]
        required_here = min(minimum, count)
        reserve = sum(min(minimum, int(config["split"][f"normal_{later}"])) for later in names[index + 1:])
        if len(boundary) < required_here + reserve:
            raise ValueError("not enough verified boundary examples to cover all requested splits")
        requested = max(required_here, round(count * share))
        preferred = take(boundary, min(requested, len(boundary) - reserve), rng, forbidden, name)
        forbidden.update(normalize(row["prompt"]) for row in preferred)
        ordinary = [row for row in pools["normal"] if not row.get("boundary_normal") and normalize(row["prompt"]) not in forbidden]
        rest = take(ordinary, min(count - len(preferred), len(ordinary)), rng, forbidden, name)
        forbidden.update(normalize(row["prompt"]) for row in rest)
        missing = count - len(preferred) - len(rest)
        remaining_boundary = [row for row in boundary if normalize(row["prompt"]) not in forbidden]
        if missing > len(remaining_boundary) - reserve:
            raise ValueError(f"{name}: insufficient verified normal examples while reserving later split coverage")
        extra = take(remaining_boundary, missing, rng, forbidden, name)
        forbidden.update(normalize(row["prompt"]) for row in extra)
        combined = bad + preferred + rest + extra
        rng.shuffle(combined)
        splits[name] = combined
    return splits


def main() -> None:
    # One audited pipeline; an unaudited prepare_data shortcut must not make a
    # training-ready manifest with request labels inferred from response labels.
    from .audit_external import main as audit_main

    audit_main()


if __name__ == "__main__":
    main()
