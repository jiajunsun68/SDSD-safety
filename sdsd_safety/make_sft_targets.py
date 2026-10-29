"""Build refusal targets for the safety-SFT baseline.

The audited PKU external responses are mostly safe alternative answers, not
refusals.  A pilot SFT on them lowered dev safety (99.7% -> 90.0%) and GSM8K
(90.5 -> 71.0), so it is not a fair safety-SFT baseline.  Here the base model is
asked, under the OPSA harmful system prompt, to refuse each harmful training
prompt.  A generated response is kept only if Qwen3Guard marks it Safe and the
response is a refusal; otherwise a fixed brief refusal is used.  Normal prompts
keep the audited alpaca responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .judging import Qwen3GuardJudge
from .metrics import is_refusal
from .modeling import generate_rollouts, load_base_model, load_tokenizer
from .train import OPSA_HARMFUL_PROMPT
from .utils import load_config, read_jsonl, resolve_path, write_json, write_jsonl

FALLBACK_REFUSAL = (
    "I'm sorry, but I can't help with that. "
    "If you have a different question, I'm happy to help."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_jsonl(resolve_path(config, config["data"]["train"]))
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
    for row, response, judgment in zip(harmful, responses, judgments, strict=True):
        refusal = judgment.get("refusal")
        if refusal is None:
            refusal = is_refusal(response)
        if judgment["safety"] == "Safe" and refusal:
            targets[row["id"]] = (response, "self_refusal")
            self_refusals += 1
        else:
            targets[row["id"]] = (FALLBACK_REFUSAL, "fallback_refusal")
            fallbacks += 1
        print(f"target {len(targets)}/{len(harmful)}", flush=True)

    output = []
    for row in rows:
        row = dict(row)
        if row["kind"] == "harmful":
            row["sft_response"], row["sft_source"] = targets[row["id"]]
        else:
            row["sft_response"] = row["external_response"]
            row["sft_source"] = "alpaca_external"
        output.append(row)
    path = Path(config["project_root"]) / "data" / "train_sft.jsonl"
    write_jsonl(path, output)
    stats = {
        "rows": len(output),
        "harmful": len(harmful),
        "self_refusals": self_refusals,
        "fallbacks": fallbacks,
        "self_refusal_rate": self_refusals / max(len(harmful), 1),
        "path": str(path),
    }
    write_json(Path(config["project_root"]) / "data" / "train_sft_manifest.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
