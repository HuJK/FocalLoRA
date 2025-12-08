#!/usr/bin/env python3
"""
Utility to evaluate a base model (and optional LoRA adapter) on the MMLU benchmark.

The script mirrors the loading/generation settings used in `_testmodel.py` so the
results are comparable. Pass explicit `--model_path` / `--lora_path` arguments or
set the MODEL_PATH / LORA_PATH environment variables.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
import tqdm
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "../models/Llama-3.1-8B-Instruct/"
DEFAULT_LORA = "../LoraAdapter/Llama-3.1-8B-Instruct_modified_0.01/batch_0"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert tutor. Answer multiple choice questions by returning only the "
    "single letter (A, B, C, or D) for the best option. Do not add justification."
)
CHOICE_LETTERS = ["A", "B", "C", "D"]
CHOICE_PATTERN = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an MMLU evaluation for a base model and optional LoRA adapter.")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--lora_path", default=os.environ.get("LORA_PATH", DEFAULT_LORA))
    parser.add_argument(
        "--subjects",
        type=str,
        default=os.environ.get("MMLU_SUBJECTS", "all"),
        help="Comma-separated list of MMLU subjects/configs (default: all).",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test", "train"],
        default=os.environ.get("MMLU_SPLIT", "test"),
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=int(os.environ.get("MMLU_MAX_SAMPLES", "0")),
        help="Optional cap on the number of questions per subject (0 means all).",
    )
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int,
                        default=min(int(os.environ.get("MAX_NEW_TOKENS", "16")), 32))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cuda_device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--attn_impl", default="eager", choices=["eager", "flash_attention_2"])
    return parser.parse_args()


def normalize_subjects(value: str) -> List[str]:
    bits = [part.strip() for part in (value or "").split(",")]
    subjects = [part for part in bits if part]
    return subjects or ["all"]


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


def load_mmlu_subjects(subjects: Sequence[str], split: str, max_samples: int):
    subject_sets: List[Tuple[str, Iterable[Dict]]] = []
    for subject in subjects:
        print(f"Loading MMLU subject '{subject}' ({split} split)...")
        dataset = load_dataset("cais/mmlu", subject, split=split)
        if max_samples and max_samples > 0:
            sample_count = min(max_samples, len(dataset))
            dataset = dataset.select(range(sample_count))
        subject_sets.append((subject, dataset))
    return subject_sets


def build_mmlu_prompt(tokenizer, system_prompt: str, subject: str, question: str, choices: Sequence[str]) -> str:
    choice_lines = [f"{CHOICE_LETTERS[idx]}. {choice}" for idx, choice in enumerate(choices)]
    user_message = "\n".join([
        f"Subject: {subject}",
        f"Question: {question.strip()}",
        "Choices:",
        *choice_lines,
        "Answer with only the single letter (A, B, C, or D).",
    ])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str) -> str:
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    torch_device = torch.device(device)
    encoded = {k: v.to(torch_device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoded["attention_mask"].sum(dim=1).tolist()[0]
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def extract_choice_letter(response: str) -> str:
    if not response:
        return ""
    match = CHOICE_PATTERN.search(response)
    if match:
        return match.group(1).upper()
    response = response.strip().upper()
    if response and response[0] in CHOICE_LETTERS:
        return response[0]
    return ""


def letter_for_answer_idx(idx: int) -> str:
    if 0 <= idx < len(CHOICE_LETTERS):
        return CHOICE_LETTERS[idx]
    raise ValueError(f"Unexpected MMLU answer index: {idx}")


def evaluate_model(
    model_label: str,
    model,
    tokenizer,
    subject_sets: Sequence[Tuple[str, Iterable[Dict]]],
    args,
    device: str,
):
    total = 0
    correct = 0
    no_parse = 0
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    start_time = time.time()
    for configured_subject, dataset in subject_sets:
        for idx, example in tqdm.tqdm(enumerate(dataset)):
            subject = example.get("subject", configured_subject)
            prompt = build_mmlu_prompt(tokenizer, args.system_prompt, subject, example["question"], example["choices"])
            response = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, device)
            predicted = extract_choice_letter(response)
            gold = letter_for_answer_idx(int(example["answer"]))

            total += 1
            entry = per_subject[subject]
            entry["total"] += 1

            if not predicted:
                no_parse += 1
            elif predicted == gold:
                correct += 1
                entry["correct"] += 1
            if args.max_samples and idx + 1 >= args.max_samples:
                break

    elapsed = time.time() - start_time
    accuracy = (correct / total) * 100 if total else 0.0

    print(f"\n=== {model_label} ===")
    print(f"Questions evaluated : {total}")
    print(f"Accuracy            : {accuracy:.2f}% ({correct}/{total})")
    if no_parse:
        print(f"Unparsed responses  : {no_parse}")
    print(f"Elapsed time        : {elapsed:.1f}s")
    print("Per-subject accuracy:")
    for subject, stats in sorted(per_subject.items()):
        subject_acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
        print(f"  {subject:30s} {stats['correct']:4d}/{stats['total']:4d} ({subject_acc:5.2f}%)")


def main():
    args = parse_args()
    args.max_new_tokens = max(1, min(args.max_new_tokens, 32))
    subjects = normalize_subjects(args.subjects)

    cuda_spec = (args.cuda_device or "").strip()
    if cuda_spec:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_spec
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model from: {args.model_path}")
    tokenizer, model = load_tokenizer_and_model(args.model_path, attn_impl=args.attn_impl)

    subject_sets = load_mmlu_subjects(subjects, args.split, args.max_samples)
    evaluate_model("Base model", model, tokenizer, subject_sets, args, device)

    lora_path = (args.lora_path or "").strip()
    if lora_path:
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        print(f"\nApplying LoRA adapter from: {lora_path}")
        tuned_model = PeftModel.from_pretrained(model, lora_path, device_map="auto")
        evaluate_model("LoRA-tuned model", tuned_model, tokenizer, subject_sets, args, device)
    else:
        print("\n[Info] No LoRA path provided; skipping adapter evaluation.")


if __name__ == "__main__":
    main()
