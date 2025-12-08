#!/usr/bin/env python3
"""
Quick utility to compare a base model against a LoRA adapter on a single chat prompt.

The prompt is assembled from three pieces:
  * COND1 -> system message
  * COND2 -> prepended instruction in the user message (used for conflicts)
  * TASK  -> the actual task/question

The values can come from CLI flags, environment variables, or the built-in defaults.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import PeftModel


DEFAULT_MODEL = "../models/Llama-3.1-8B-Instruct/"
DEFAULT_LORA = "../LoraAdapter/Llama-3.1-8B-Instruct_modified_0.01/batch_0"
DEFAULT_COND1 = "Your entire response should be in English, no other language is allowed."
DEFAULT_COND2 = "Your entire response should be in French, no other language is allowed."
DEFAULT_TASK = (
    "Describe the greenhouse effect and explain how human activities, "
    "such as fossil-fuel combustion, intensify this natural process."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the base model output vs. a LoRA adapter.")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--lora_path", default=os.environ.get("LORA_PATH", DEFAULT_LORA))
    parser.add_argument("--cond1", default=os.environ.get("COND1", DEFAULT_COND1))
    parser.add_argument("--cond2", default=os.environ.get("COND2", DEFAULT_COND2))
    parser.add_argument("--task", default=os.environ.get("TASK", DEFAULT_TASK))
    parser.add_argument("--max_new_tokens", type=int,
                        default=min(int(os.environ.get("MAX_NEW_TOKENS", "512")), 512))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cuda_device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--attn_impl", default="eager", choices=["eager", "flash_attention_2"])
    return parser.parse_args()


def load_tokenizer_and_model(model_path: str, attn_impl: str):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, cond1: str, cond2: str, task: str) -> str:
    cond1 = (cond1 or "").strip()
    cond2 = (cond2 or "").strip()
    task = (task or "").strip()
    if not cond1:
        raise ValueError("COND1/system message cannot be empty.")
    user_bits: List[str] = [x for x in (cond2, task) if x]
    user_message = " ".join(user_bits).strip()
    messages = [{"role": "system", "content": cond1}]
    if user_message:
        messages.append({"role": "user", "content": user_message})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str) -> str:
    tokenized = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    torch_device = torch.device(device)
    tokenized = {k: v.to(torch_device) for k, v in tokenized.items()}
    with torch.no_grad():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    attention_mask = tokenized["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    generated_tokens = outputs[0][prompt_lengths[0]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_cases(model_label: str, model, tokenizer, prompts, max_new_tokens: int, temperature: float, device: str):
    for case_label, prompt in prompts:
        print(f"\n=== {model_label}, {case_label} ===")
        print("\nPrompt:\n")
        print(prompt)
        print("\nOutput:\n")
        response = generate(model, tokenizer, prompt, max_new_tokens, temperature, device)
        print(response)


def main():
    args = parse_args()
    args.max_new_tokens = max(1, min(args.max_new_tokens, 512))
    cuda_spec = (args.cuda_device or "").strip()
    if cuda_spec:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_spec
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model from: {args.model_path}")
    tokenizer, model = load_tokenizer_and_model(args.model_path, attn_impl=args.attn_impl)
    normal_prompt = build_prompt(tokenizer, args.cond1, "", args.task)
    conflict_prompt = build_prompt(tokenizer, args.cond1, args.cond2, args.task)
    prompt_cases = [
        ("normal case", normal_prompt),
        ("conflict case", conflict_prompt),
    ]

    run_cases("base model", model, tokenizer, prompt_cases, args.max_new_tokens, args.temperature, device)

    lora_path = (args.lora_path or "").strip()
    if lora_path:
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        print(f"\nApplying LoRA adapter from: {lora_path}")
        lora_model = PeftModel.from_pretrained(model, lora_path, device_map="auto")
        run_cases("tuned model", lora_model, tokenizer, prompt_cases, args.max_new_tokens, args.temperature, device)
    else:
        print("\n[Info] No LoRA path provided; skipping adapter comparison.")


if __name__ == "__main__":
    main()
