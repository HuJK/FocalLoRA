# coding: utf-8

import os, json, argparse
from pathlib import Path
from functools import lru_cache
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel  # LoRA support

def find_subsequence(full, sub):
    n, m = len(full), len(sub)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if full[i : i + m] == sub:
            return i
    return -1

@lru_cache(maxsize=None)
def clean_token(tok: str) -> str:
    return tok.lstrip("▁")

def build_token_ranges(tokenizer, full_ids, sys_ids, usr_ids):
    # First try exact sub-sequence match
    s0 = find_subsequence(full_ids, sys_ids)
    if s0 != -1:
        s1 = s0 + len(sys_ids) - 1
        if usr_ids:
            u0 = find_subsequence(full_ids, usr_ids)
            u1 = u0 + len(usr_ids) - 1 if u0 != -1 else None
            usr_range = (u0, u1) if u0 != -1 else None
        else:
            usr_range = None
        return (s0, s1), usr_range

    # Try known chat templates
    tid = tokenizer.convert_tokens_to_ids
    start_header = tid("<|start_header_id|>")
    end_header   = tid("<|end_header_id|>")
    eot_id       = tokenizer.eos_token_id
    sys_tok      = tid("<|system|>")
    end_tok      = tid("<|end|>")
    im_start     = tid("<|im_start|>")
    im_end       = tid("<|im_end|>")
    inst_start   = tid("[INST]")
    inst_end     = tid("[/INST]")
    row = full_ids

    def find_token_range(row, start_token, end_token, start_offset=1):
        try:
            s = row.index(start_token) + start_offset
            e = row.index(end_token, s)
            return s, e
        except ValueError:
            return None

    if start_header in row:
        try:
            s = row.index(end_header) + 1
            e = row.index(eot_id)
            return (s, e), None
        except ValueError:
            pass

    if sys_tok in row:
        try:
            s = row.index(sys_tok) + 1
            e = row.index(end_tok, s)
            return (s, e), None
        except ValueError:
            pass

    if im_start in row and im_end in row:
        for pos in [i for i, t in enumerate(row) if t == im_start]:
            if pos + 1 < len(row) and tokenizer.decode([row[pos + 1]]).strip() == "system":
                s = pos + 2
                try:
                    e = row.index(im_end, s)
                    return (s, e), None
                except ValueError:
                    pass

    if inst_start in row and inst_end in row:
        try:
            ist = row.index(inst_start) + 1
            iend = row.index(inst_end)
            split = None
            for i in range(ist, iend - 1):
                if row[i] == eot_id and row[i + 1] == eot_id:
                    split = i
                    break
            if split is None:
                for i in range(ist, iend):
                    if tokenizer.decode([row[i]]).isspace():
                        split = i
                        break
            if split and ist < split:
                return (ist, split), (split + 1, iend)
            else:
                return (ist, iend), None
        except ValueError:
            pass

    # Fallback
    s0 = find_subsequence(row, sys_ids)
    s1 = s0 + len(sys_ids) - 1 if s0 != -1 else -1
    u0 = find_subsequence(row, usr_ids) if usr_ids else -1
    u1 = u0 + len(usr_ids) - 1 if u0 != -1 else -1
    sys_range = (s0, s1) if s0 != -1 else None
    usr_range = (u0, u1) if u0 != -1 else None
    return sys_range, usr_range

def load_important_heads(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for tag, _ in data["important_heads"]:
        l = int(tag.split("_")[0][1:])
        h = int(tag.split("_")[1][1:])
        pairs.append((l, h))
    return pairs, [tag for tag, _ in data["important_heads"]]

def last_token_selected_heads(attentions, selected_pairs):
    S = attentions[0].shape[-1]
    last = S - 1
    rows = []
    for (l, h) in selected_pairs:
        vec = attentions[l][0, h, last, :].to(torch.float32).cpu().numpy()
        rows.append(vec)
    return np.stack(rows)

def average_heads_last_token(attentions):
    L = len(attentions)
    S = attentions[0].shape[-1]
    last = S - 1
    mat = np.zeros((L, S), dtype=np.float32)
    for l, attn in enumerate(attentions):
        mat[l] = attn[0, :, last, :].mean(dim=0).to(torch.float32).cpu().numpy()
    return mat

def plot_heatmap(mat, tokens, row_labels, out_path, title):
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_red", ["#FEFFDA", "#CC3F39"], N=256)
    cbar_font = {'size': 18}
    xtick_font = {'fontsize': 10}
    ytick_font = {'fontsize': 10}

    plt.figure(figsize=(max(6, mat.shape[0] * 0.6), max(4, len(tokens) * 0.35)))
    ax = sns.heatmap(
        mat.T,
        cmap=custom_cmap,
        vmin=0.0,
        vmax=0.2,
        xticklabels=row_labels,
        yticklabels=[clean_token(t) for t in tokens],
        cbar_kws={"label": "Attention Score", "format": '%.2f'}
    )
    ax.set_xlabel("Important Heads (x)")
    ax.set_ylabel("Input Tokens (y)")
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', labelsize=xtick_font["fontsize"])
    ax.tick_params(axis='y', labelsize=ytick_font["fontsize"])
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_font["size"])
    cbar.set_label("Attention Score", fontsize=cbar_font["size"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved heatmap to: {out_path}")


def main(args):
    json_file = Path(args.json_file)
    if not json_file.exists():
        raise RuntimeError(f"JSON file not found: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Expected a list of samples in the JSON file.")

    if len(args.cuda) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.cuda))
        device = None
    else:
        device = torch.device(f"cuda:{args.cuda[0]}" if torch.cuda.is_available() else "cpu")

    print(f"🔵 Loading base model from {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else {"": device.index},
        trust_remote_code=True,
        attn_implementation="eager"
    )

    if args.lora_path and args.lora_path.strip() != "":
        print(f"🟣 Applying LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            device_map="auto" if device is None else {"": device.index}
        )
    else:
        print("⚪️ No LoRA adapter applied, using base model only.")

    model.eval()
    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    for sample in tqdm(samples, desc="Processing Samples"):
        sys_msg = sample["system_message"]
        usr_msg = sample.get("user_message", "") or ""
        messages = [{"role": "system", "content": sys_msg}]
        if usr_msg.strip():
            messages.append({"role": "user", "content": usr_msg})

        with torch.no_grad():
            chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(chat_input, return_tensors="pt")
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}

        sys_ids = tokenizer(sys_msg, add_special_tokens=False)["input_ids"]
        usr_ids = tokenizer(usr_msg, add_special_tokens=False)["input_ids"] if usr_msg.strip() else []
        full_ids = inputs["input_ids"][0].tolist()
        sys_range, usr_range = build_token_ranges(tokenizer, full_ids, sys_ids, usr_ids)
        s0, s1 = sys_range

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        mat = average_heads_last_token(outputs.attentions)
        tokens = tokenizer.convert_ids_to_tokens(full_ids)
        wanted_idx = list(range(s0, s1 + 1))
        if usr_range:
            u0, u1 = usr_range
            wanted_idx += list(range(u0, u1 + 1))
        sub_mat = mat[:, wanted_idx]
        sub_tokens = [tokens[i] for i in wanted_idx]

        sample_id = sample.get('id', 'unknown')
        title = f"Last-Token → System/User Tokens  (sample id: {sample_id})"

        if args.important_file and os.path.exists(args.important_file):
            selected_pairs, head_tags = load_important_heads(args.important_file)
            print("✔ Loaded important heads:", head_tags)
            use_all_heads = False
        else:
            selected_pairs, head_tags = [], []
            print("⚠ No important_heads.json found, visualizing average over all heads")
            use_all_heads = True

        row_labels = head_tags if not use_all_heads else [f"L{l}" for l in range(mat.shape[0])]
        plot_heatmap(sub_mat, sub_tokens, row_labels, os.path.join(out_dir, f"{sample_id}_attn_map.png"), title)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        mat_all = last_token_selected_heads(outputs.attentions, selected_pairs)
        sub_mat2 = mat_all[:, wanted_idx]
        sub_tokens2 = [tokens[i] for i in wanted_idx]
        title = " "
        plot_heatmap(sub_mat2, sub_tokens2, head_tags, os.path.join(out_dir, f"{sample_id}_imp_heads.png"), title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--important_file", type=str, default="important_heads.json",
                        help="Path to important_heads.json file with selected attention heads.")
    parser.add_argument("--model_path", type=str, default="/home/user/models/Llama-3-8B",
                        help="Path to base pretrained model.")
    parser.add_argument("--lora_path", type=str, default="", help="Optional LoRA adapter path.")
    parser.add_argument("--json_file", type=str, default="samples.json", help="Input JSON file.")
    parser.add_argument("--output_path", type=str, default="./attn_vis", help="Output folder for heatmaps.")
    parser.add_argument("--cuda", type=int, nargs='+', default=[0],
                        help="CUDA device indices, e.g. 0 or 0 1.")
    args = parser.parse_args()
    main(args)
