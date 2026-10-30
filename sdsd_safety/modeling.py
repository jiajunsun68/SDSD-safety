"""Model loading, LoRA, chat formatting, rollout generation, and token scoring."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(path: str, dtype: str = "bfloat16", attn_implementation: str = "sdpa", device: str = "cuda"):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    return model


def attach_lora(model, cfg: dict[str, Any]):
    lora = LoraConfig(
        r=int(cfg["r"]),
        lora_alpha=int(cfg["alpha"]),
        lora_dropout=float(cfg["dropout"]),
        target_modules=list(cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def lora_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if "lora_" in name}


def copy_lora(source, target) -> None:
    source_params = lora_parameters(source)
    target_params = lora_parameters(target)
    if set(source_params) != set(target_params):
        raise RuntimeError("student/teacher LoRA parameter names differ")
    with torch.no_grad():
        for name, param in source_params.items():
            target_params[name].copy_(param)


@torch.no_grad()
def ema_lora(source, target, decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must lie in [0, 1]")
    source_params = lora_parameters(source)
    target_params = lora_parameters(target)
    for name, param in source_params.items():
        target_params[name].mul_(decay).add_(param, alpha=1.0 - decay)


def chat_prompt(tokenizer, prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def teacher_prompt(tokenizer, prompt: str, demo: str | None, max_prompt_tokens: int = 512) -> str:
    """Place a reference in context while keeping the original user request intact.

    Only the reference may be shortened to meet the context budget. A question
    that does not fit is rejected instead of silently losing its first tokens.
    """
    if demo is None:
        text = chat_prompt(tokenizer, prompt)
        _prompt_ids(tokenizer, text, max_prompt_tokens)
        return text

    def format_reference(reference: str) -> str:
        system = (
            "A reference response is provided as context. Answer the user's request directly; "
            "do not mention this reference or these instructions.\n\nReference response:\n"
            + reference
        )
        return chat_prompt(tokenizer, prompt, system=system)

    _prompt_ids(tokenizer, chat_prompt(tokenizer, prompt), max_prompt_tokens)
    text = format_reference(demo)
    if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_prompt_tokens:
        return text
    demo_ids = tokenizer(demo, add_special_tokens=False)["input_ids"]
    low, high = 0, len(demo_ids)
    best = None
    while low <= high:
        count = (low + high) // 2
        reference = tokenizer.decode(demo_ids[:count], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        candidate = format_reference(reference)
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) <= max_prompt_tokens:
            if reference.strip():
                best = candidate
            low = count + 1
        else:
            high = count - 1
    if best is None:
        raise ValueError("max_prompt_tokens leaves no room for a teacher reference after preserving the question")
    return best


def trim_at_eos(ids: list[int], eos_token_id: int) -> list[int]:
    if eos_token_id in ids:
        return ids[: ids.index(eos_token_id) + 1]
    return ids


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompts: list[str],
    num_rollouts: int = 4,
    max_new_tokens: int = 192,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    greedy: bool = False,
    system: str | None = None,
    max_prompt_tokens: int = 512,
) -> list[list[dict[str, Any]]]:
    """Return per-prompt lists of {text, token_ids} for `num_rollouts` samples."""
    model.eval()
    tokenizer.padding_side = "left"
    old_cache = model.config.use_cache
    model.config.use_cache = True
    results: list[list[dict[str, Any]]] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            texts = [chat_prompt(tokenizer, prompt, system=system) for prompt in chunk]
            encoded = tokenizer(
                texts,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            if encoded["input_ids"].shape[1] > max_prompt_tokens:
                raise ValueError(
                    f"question context exceeds max_prompt_tokens={max_prompt_tokens}; "
                    "increase the explicit budget instead of truncating the request"
                )
            kwargs: dict[str, Any] = {
                **encoded,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": not greedy,
                "num_return_sequences": num_rollouts,
            }
            if not greedy:
                kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": 0})
            output = model.generate(**kwargs)
            prompt_width = encoded["input_ids"].shape[1]
            completion_ids = output[:, prompt_width:]
            texts_out = tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            flat_ids = completion_ids.tolist()
            for index in range(len(chunk)):
                samples = []
                for j in range(num_rollouts):
                    position = index * num_rollouts + j
                    ids = trim_at_eos(flat_ids[position], tokenizer.eos_token_id)
                    samples.append({"text": texts_out[position], "token_ids": ids})
                results.append(samples)
    finally:
        model.config.use_cache = old_cache
    return results


def _prompt_ids(tokenizer, prompt_text: str, max_prompt_tokens: int = 512) -> list[int]:
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_prompt_tokens:
        raise ValueError(f"context has {len(ids)} tokens, exceeding max_prompt_tokens={max_prompt_tokens}")
    return ids


def completion_logits(
    model,
    tokenizer,
    prompt_texts: list[str],
    completion_ids: list[list[int]],
    micro_batch: int = 4,
    with_grad: bool = False,
    max_prompt_tokens: int = 512,
) -> list[torch.Tensor]:
    """Logits predicting each completion token, aligned per row.

    Returns a list of [Lc, vocab] tensors.  The model must be in eval mode for
    teacher/reference scoring; for the student pass with_grad=True.
    """
    outputs: list[torch.Tensor] = []
    context = torch.enable_grad() if with_grad else torch.no_grad()
    model.config.use_cache = False
    with context:
        for start in range(0, len(prompt_texts), micro_batch):
            prompt_chunk = prompt_texts[start : start + micro_batch]
            completion_chunk = completion_ids[start : start + micro_batch]
            sequences = []
            comp_lens = []
            for prompt_text, comp in zip(prompt_chunk, completion_chunk, strict=True):
                prompt_ids = _prompt_ids(tokenizer, prompt_text, max_prompt_tokens)
                sequences.append(prompt_ids + list(comp))
                comp_lens.append(len(comp))
            max_len = max(len(sequence) for sequence in sequences)
            max_comp = max(comp_lens)
            keep = max_comp + 1
            input_ids = torch.full((len(sequences), max_len), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
            attention = torch.zeros((len(sequences), max_len), dtype=torch.long, device=model.device)
            for row, sequence in enumerate(sequences):
                input_ids[row, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long, device=model.device)
                attention[row, -len(sequence) :] = 1
            position_ids = attention.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention == 0, 0)
            logits = model(
                input_ids=input_ids, attention_mask=attention,
                position_ids=position_ids, logits_to_keep=keep,
            ).logits
            if logits.shape[1] != keep:
                raise RuntimeError(f"logits_to_keep returned {logits.shape[1]} positions, expected {keep}")
            for row, comp_len in enumerate(comp_lens):
                start_index = keep - comp_len - 1
                outputs.append(logits[row, start_index : keep - 1])
    return outputs


def sampled_logprobs(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def encode_sft_pair(tokenizer, prompt: str, response: str, max_length: int) -> dict[str, Any]:
    prompt_text = chat_prompt(tokenizer, prompt)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat prefix mismatch for SFT example")
    if len(prompt_ids) >= max_length:
        raise ValueError("SFT context leaves no supervised completion tokens")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "prompt_tokens": len(prompt_ids)}
