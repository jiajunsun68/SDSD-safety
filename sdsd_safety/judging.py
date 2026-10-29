"""Local safety judging with Qwen3Guard-Gen-4B (no external API)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import parse_qwen3guard


class Qwen3GuardJudge:
    def __init__(self, model_path: str, batch_size: int = 32, max_new_tokens: int = 32, device: str = "cuda"):
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def _generate(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in messages_list
        ]
        outputs: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            generated = self.model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            continuation = generated[:, batch["input_ids"].shape[1] :]
            outputs.extend(
                self.tokenizer.batch_decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )
        return outputs

    def judge_responses(self, prompts: list[str], responses: list[str]) -> list[dict[str, Any]]:
        messages = [
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            for prompt, response in zip(prompts, responses, strict=True)
        ]
        raw = self._generate(messages)
        rows = []
        for text in raw:
            safety, refusal = parse_qwen3guard(text)
            rows.append({"safety": safety, "refusal": refusal, "raw": text})
        return rows

    def judge_prompts(self, prompts: list[str]) -> list[dict[str, Any]]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        raw = self._generate(messages)
        rows = []
        for text in raw:
            safety, _ = parse_qwen3guard(text)
            rows.append({"safety": safety, "raw": text})
        return rows

    @staticmethod
    def is_safe(row: dict[str, Any]) -> bool:
        return row.get("safety") == "Safe"


def judge_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    safe = sum(row["safety"] == "Safe" for row in rows)
    unsafe = sum(row["safety"] == "Unsafe" for row in rows)
    controversial = sum(row["safety"] == "Controversial" for row in rows)
    unparsed = total - safe - unsafe - controversial
    return {
        "rows": total,
        "safe_rate": safe / total if total else 0.0,
        "unsafe_rate": unsafe / total if total else 0.0,
        "controversial_rate": controversial / total if total else 0.0,
        "parse_failure_rate": unparsed / total if total else 0.0,
    }
