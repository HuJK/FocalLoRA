# -*- coding: utf-8 -*-
"""
collect_head_attn_dicts.py
==========================
Extract attention vectors from normal/conflict samples and produce two dictionaries:
    normal_attns : { "L3_H5": [np.ndarray, ...], ... }
    conflict_attns : same structure
The result is saved as .npz or .pkl, for direct use by visualize_head_importance.py.
"""

import os, json, argparse
from pathlib import Path
from collections import defaultdict

import torch, numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM
)

# ------------- A. General model loading -------------
def load_model(model_dir: str, device):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        device_map="auto" if device is None else {"": device.index},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tok

# ------------- B. Extract attention for one sample -------------
@torch.inference_mode()
def get_last_token_attn(model, tok, sys_msg: str, user_msg: str):
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ]
    text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(text_in, return_tensors="pt").to(model.device)

    out = model(**input_ids, output_attentions=True)
    attn = out.attentions  # list[n_layers] of tuple(batch, n_head, tgt, src)

    vecs = []  # For each layer and head, extract the last token row
    for layer_id, layer_attn in enumerate(attn):
        A = layer_attn[0]
        last_row = A[:, -1, :].to(torch.float32).cpu().numpy()
        vecs.append(last_row)
    return vecs  # list of n_layers, each [n_head, src_len]

def save_important_heads_json(normal_attns, conflict_attns, out_json, top_k=10):
    scores = score_heads_by_tracker_method(normal_attns, conflict_attns)
    sorted_heads = sorted(scores.items(), key=lambda x: -x[1])
    top_heads = sorted_heads[:top_k]

    output = {
        "important_heads": [[k, float(v)] for k, v in top_heads]
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"📄 Saved important heads to {out_json}")

# ------------- C. Main extraction logic -------------
def collect_dicts(json_path: str, model, tok):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group samples by ID prefix, split into normal/conflict pairs
    grouped = defaultdict(lambda: {"normal": None, "conflict": None})
    for sample in data:
        base = sample["id"].replace("_normal", "").replace("_conflict", "")
        grouped[base][sample["label"]] = sample

    normal_attns = defaultdict(list)
    conflict_attns = defaultdict(list)

    for base_id, pair in tqdm(grouped.items(), desc="Collecting attention"):
        for label in ["normal", "conflict"]:
            samp = pair[label]
            if samp is None:
                continue

            user_msg = f"{samp['task']} {samp['user_message']}".strip() if samp["user_message"].strip() else samp["task"]
            sys_msg  = samp["system_message"]

            vecs = get_last_token_attn(model, tok, sys_msg, user_msg)

            for L, layer_vec in enumerate(vecs):
                n_head = layer_vec.shape[0]
                for H in range(n_head):
                    key = f"L{L}_H{H}"
                    if label == "normal":
                        normal_attns[key].append(layer_vec[H])
                    else:
                        conflict_attns[key].append(layer_vec[H])

    return normal_attns, conflict_attns

# ------------- D. Saving utilities -------------
def save_dicts(normal_attns, conflict_attns, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        normal=normal_attns,
                        conflict=conflict_attns)
    print(f"✅ Saved attention dicts to {out_path}")

# ------------- E. Command-line interface -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path",  required=True, help="Input JSON with normal/conflict samples")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model")
    parser.add_argument("--cuda",       type=int, nargs="+", default=[0], help="CUDA device ID(s)")
    parser.add_argument("--out_file",   default="head_attn_dicts.npz", help="Output file (.npz)")
    args = parser.parse_args()

    # Device setup
    if len(args.cuda) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda))
        device = None
    else:
        device = torch.device(f"cuda:{args.cuda[0]}" if torch.cuda.is_available() else "cpu")

    model, tok = load_model(args.model_path, device)

    normal_attns, conflict_attns = collect_dicts(args.json_path, model, tok)

    out_json = Path(args.out_file).with_name("important_heads.json")
    save_important_heads_json(normal_attns, conflict_attns, out_json)

if __name__ == "__main__":
    main()
