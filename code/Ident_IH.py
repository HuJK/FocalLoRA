# -*- coding: utf-8 -*-
"""
This script loads a specified LLM model, computes head-wise attention scores
for normal vs. conflict instruction samples, and generates multiple heatmap visualizations.
It outputs both per-sample attention maps and average attention patterns,
highlighting the most discriminative attention heads.
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
)
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from pathlib import Path

def load_llama3_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index if device.type == "cuda" else "cpu"},
        trust_remote_code=True,
        attn_implementation="eager"
    )
    return model, tokenizer

def compute_stability_separation_score(normal_scores, conflict_scores, epsilon=1e-6):
    scores = {}
    for key in normal_scores:
        mu_n, std_n = np.mean(normal_scores[key]), np.std(normal_scores[key])
        mu_a, std_a = np.mean(conflict_scores[key]), np.std(conflict_scores[key])
        score = abs(mu_n - mu_a) / (std_n + std_a + epsilon)
        scores[key] = score
    return scores

def get_attn_lh(attentions, instr_start, instr_end):
    n_layers = len(attentions)
    n_heads = attentions[0][0].shape[0]
    last_token_idx = attentions[0][0].shape[2] - 1
    attn_lh = {}
    for l in range(n_layers):
        for h in range(n_heads):
            row = attentions[l][0][h, last_token_idx, :].to(torch.float32).detach().cpu().numpy()
            score = np.sum(row[instr_start:instr_end])
            attn_lh[f"L{l}_H{h}"] = score
    return attn_lh

def generate_global_attention_heatmaps(attentions, tokenizer, input_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    last_token_idx = attentions[0][0].shape[2] - 1
    n_layers = len(attentions)
    n_heads = attentions[0][0].shape[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.replace("▁", "") if "▁" in t else t for t in tokens]

    heads_layers_mat = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            heads_layers_mat[l, h] = attentions[l][0][h, last_token_idx, :].mean().item()

    plt.figure(figsize=(n_heads * 0.4, n_layers * 0.4))
    sns.heatmap(heads_layers_mat, cmap="viridis", xticklabels=[f"H{h}" for h in range(n_heads)],
                yticklabels=[f"L{l}" for l in range(n_layers)], annot=True, fmt=".2f")
    plt.title("Global Heads-Layers Attention")
    plt.savefig(os.path.join(output_dir, "global_heads_layers_attention.png"), dpi=300, bbox_inches='tight')
    plt.close()

    mat = np.zeros((n_layers, len(tokens)))
    for l in range(n_layers):
        avg = attentions[l][0][:, last_token_idx, :].mean(dim=0).to(torch.float32).cpu().numpy()
        mat[l, :] = avg

    plt.figure(figsize=(len(tokens) * 0.5, n_layers * 0.5))
    sns.heatmap(mat, xticklabels=tokens, yticklabels=[f"L{l}" for l in range(n_layers)], cmap="viridis", annot=False)
    plt.xticks(rotation=90)
    plt.title("Global Layers → Tokens (Last Token)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_layers_tokens_attention.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_heads_token_heatmap(attentions, important_heads, tokenizer, input_ids, output_dir, prefix=""):
    last_token_idx = attentions[0][0].shape[2] - 1
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.replace("▁", "") if "▁" in t else t for t in tokens]

    for head_str, _ in important_heads:
        try:
            layer_idx = int(head_str.split("_")[0][1:])
            head_idx = int(head_str.split("_")[1][1:])
        except:
            continue

        row = attentions[layer_idx][0][head_idx, last_token_idx, :].to(torch.float32).detach().cpu().numpy()

        plt.figure(figsize=(len(tokens) * 0.5, 2))
        sns.heatmap([row], cmap="viridis", xticklabels=tokens, yticklabels=[head_str], cbar=True)
        plt.xticks(rotation=90)
        plt.title(f"{head_str} → Tokens (Last Token)")
        filename = f"{prefix}_head_token_heatmap_{head_str}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def generate_average_global_heatmaps(all_attns_dict, all_input_ids_dict, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not all_attns_dict:
        return

    n_layers = len(next(iter(all_attns_dict.values())))
    n_heads = all_attns_dict[next(iter(all_attns_dict))][0][0].shape[0]
    last_token_idx = all_attns_dict[next(iter(all_attns_dict))][0][0].shape[2] - 1

    max_seq_len = max(attns[0][0].shape[2] for attns in all_attns_dict.values())
    sum_heads_layers = np.zeros((n_layers, n_heads))
    sum_layers_tokens = np.zeros((n_layers, max_seq_len))
    count = 0
    tokens = None

    for key in all_attns_dict:
        attns = all_attns_dict[key]
        input_ids = all_input_ids_dict[key]
        cur_seq_len = attns[0][0].shape[2]

        heads_layers_mat = np.zeros((n_layers, n_heads))
        for l in range(n_layers):
            for h in range(n_heads):
                heads_layers_mat[l, h] = attns[l][0][h, last_token_idx, :].mean().item()
        sum_heads_layers += heads_layers_mat

        layer_token_mat = np.zeros((n_layers, max_seq_len))
        for l in range(n_layers):
            avg = attns[l][0][:, last_token_idx, :].mean(dim=0).to(torch.float32).cpu().numpy()
            layer_token_mat[l, :cur_seq_len] = avg

        sum_layers_tokens += layer_token_mat

        if tokens is None:
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            tokens = [t.replace("▁", "") if "▁" in t else t for t in tokens]

        count += 1

    mean_heads_layers = sum_heads_layers / count
    mean_layers_tokens = sum_layers_tokens / count

    plt.figure(figsize=(n_heads * 0.4, n_layers * 0.4))
    sns.heatmap(mean_heads_layers, cmap="viridis", xticklabels=[f"H{h}" for h in range(n_heads)],
                yticklabels=[f"L{l}" for l in range(n_layers)], annot=True, fmt=".2f")
    plt.title("Average Heads-Layers Attention")
    plt.savefig(os.path.join(output_dir, "average_heads_layers_attention.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(len(tokens) * 0.5, n_layers * 0.5))
    sns.heatmap(mean_layers_tokens[:, :len(tokens)], xticklabels=tokens, yticklabels=[f"L{l}" for l in range(n_layers)], cmap="viridis", annot=False)
    plt.xticks(rotation=90)
    plt.title("Average Layers → Tokens (Last Token)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_layers_tokens_attention.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_and_collect(model, tokenizer, system_msg, user_msg, instruction_range, output_dir=None):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attns = outputs.attentions
    attn_lh_scores = get_attn_lh(attns, instruction_range[0], instruction_range[1])

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if output_dir is not None:
        generate_global_attention_heatmaps(attns, tokenizer, inputs["input_ids"], os.path.join(output_dir, "global"))

    return attn_lh_scores, decoded, attns, inputs["input_ids"]

def process_json_dataset(json_path, model, tokenizer, output_json_path, model_type, output_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normal_scores = defaultdict(list)
    conflict_scores = defaultdict(list)
    results = []
    all_attns = {}
    all_input_ids = {}
    sample_dir_to_label = {}

    normal_attns = {}
    normal_input_ids = {}
    conflict_attns = {}
    conflict_input_ids = {}

    for sample in tqdm(data):
        system_msg = sample['system_message']
        user_msg = sample['user_message']
        label = sample['label']
        id_ = sample['id']

        sample_output_dir = os.path.join(output_dir, f"{id_}_sample")
        os.makedirs(sample_output_dir, exist_ok=True)

        attn_lh, output, attns, input_ids = run_and_collect(
            model, tokenizer, system_msg, user_msg, instruction_range=(0, 15), output_dir=sample_output_dir)

        all_attns[sample_output_dir] = attns
        all_input_ids[sample_output_dir] = input_ids
        sample_dir_to_label[sample_output_dir] = label

        if label == "normal":
            normal_attns[sample_output_dir] = attns
            normal_input_ids[sample_output_dir] = input_ids
        elif label == "conflict":
            conflict_attns[sample_output_dir] = attns
            conflict_input_ids[sample_output_dir] = input_ids

        for k, v in attn_lh.items():
            (normal_scores if label == "normal" else conflict_scores)[k].append(v)

        results.append({"id": id_, "label": label, "output": output})

    scores = compute_stability_separation_score(normal_scores, conflict_scores)
    important_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "important_heads": [(k, float(v)) for k, v in important_heads]}, f, indent=2, ensure_ascii=False)

    for sample_dir in all_attns:
        label = sample_dir_to_label[sample_dir]
        head_token_dir = os.path.join(sample_dir, "head_token")
        layer_token_dir = os.path.join(sample_dir, "layer_token")
        os.makedirs(head_token_dir, exist_ok=True)
        os.makedirs(layer_token_dir, exist_ok=True)

        generate_heads_token_heatmap(all_attns[sample_dir], important_heads, tokenizer, all_input_ids[sample_dir], head_token_dir, prefix=label)
        generate_layers_tokens_heatmap(all_attns[sample_dir], important_heads, tokenizer, all_input_ids[sample_dir], layer_token_dir, prefix=label)

    generate_average_global_heatmaps(normal_attns, normal_input_ids, tokenizer, os.path.join(output_dir, "average_all_sample", "normal"))
    generate_average_global_heatmaps(conflict_attns, conflict_input_ids, tokenizer, os.path.join(output_dir, "average_all_sample", "conflict"))

    print(f"✅ All processing completed. Results saved to: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="result.json")
    parser.add_argument("--llama3_local_path", type=str, default=" ")
    parser.add_argument("--cuda", type=int, nargs='+', default=[0])
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda[0]}") if torch.cuda.is_available() else torch.device("cpu")

    if args.model_type == "llama3-8b":
        model, tokenizer = load_llama3_model(args.llama3_local_path, device)
    elif args.model_type == "qwen-vl":
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, output_attentions=True, torch_dtype="auto",
            device_map={"": device.index if device.type == "cuda" else "cpu"})
    else:
        model_name = {
            "qwen-14b": "Qwen/Qwen2.5-14B-Instruct-1M",
            "qwen-math-7b": "Qwen/Qwen2.5-Math-7B-Instruct"
        }[args.model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, output_attentions=True, torch_dtype="auto",
            device_map={"": device.index if device.type == "cuda" else "cpu"})

    process_json_dataset(
        json_path=args.json_path,
        model=model,
        tokenizer=tokenizer,
        output_json_path=args.output_json,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
