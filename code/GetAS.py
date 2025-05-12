#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detect Important Attention Heads
--------------------------------
• Single-GPU: Forces model/LoRA to specified GPU; blocks non-target devices like cuda:0.
• Multi-GPU: Exposes user-specified GPUs; uses device_map="auto" for slicing.
"""

import os, json, argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch, numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

# ========================= 1. Model Loader =========================
def load_generic_model(model_dir: str,
                       device,
                       device_map_cfg: Dict):
    """
    device         : torch.device('cuda:i') or cpu
    device_map_cfg : {"": i} for single-GPU or "auto" for multi-GPU
    """
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=device_map_cfg,
    )
    return model, tokenizer

# ========================= 2. Score Function =========================
def trim_and_stack(rows: List[np.ndarray]) -> np.ndarray:
    L = min(len(r) for r in rows)
    return np.stack([r[:L] for r in rows])

def trim_to_same(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = min(a.shape[1], b.shape[1])
    return a[:, :L], b[:, :L]

def score_heads(normal: Dict[str, List[np.ndarray]],
                conflict: Dict[str, List[np.ndarray]],
                eps: float = 1e-6):
    scores = {}
    for k in normal:
        if k not in conflict:
            continue
        try:
            n = trim_and_stack(normal[k])
            c = trim_and_stack(conflict[k])
            n, c = trim_to_same(n, c)
        except Exception as e:
            print(f"⚠️ Skipped {k} (incompatible shape): {e}")
            continue
        if n.size == 0 or c.size == 0:
            continue

        frob = np.linalg.norm(n - c, ord="fro")
        mean_shift = np.mean(np.abs(n.mean(1) - c.mean(1)))

        def softmax(x):
            e = np.exp(x - x.max(-1, keepdims=True))
            return e / np.clip(e.sum(-1, keepdims=True), eps, None)

        p, q = softmax(n), softmax(c)
        kl = (p * (np.log(p + eps) - np.log(q + eps))).sum() / p.shape[0]

        scores[k] = 0.4 * frob + 0.3 * mean_shift + 0.3 * kl
    return scores

# ========================= 3. Extract Last-Token Attention =========================
@torch.inference_mode()
def extract_attention(model, tokenizer, sys_msg: str, usr_msg: str):
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": usr_msg},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outs = model(**inputs, output_attentions=True)
    gen = model.generate(**inputs, max_new_tokens=128)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=False)

    # Extract only assistant portion
    assistant_txt = decoded.split("assistant", 1)[-1].strip() if "assistant" in decoded else decoded.strip()
    return outs.attentions, inputs["input_ids"], assistant_txt

# ========================= 4. Main Detection Procedure =========================
def detect_heads(json_path: str, model, tokenizer, out_dir: str):
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    grouped = defaultdict(lambda: {"normal": None, "conflict": None})
    for s in raw:
        base = s["id"].replace("_normal", "").replace("_conflict", "")
        grouped[base][s["label"]] = s

    normal, conflict = defaultdict(list), defaultdict(list)
    responses = []

    for _, pair in tqdm(grouped.items()):
        for lbl in ("normal", "conflict"):
            sample = pair[lbl]
            if sample is None:
                continue
            usr_msg = f"{sample['task']} {sample['user_message']}".strip() if sample["user_message"].strip() else sample["task"]
            attn, ids, output = extract_attention(model, tokenizer, sample["system_message"], usr_msg)

            responses.append({
                "id": sample["id"], "label": lbl, "output": output
            })

            n_layer = len(attn)
            n_head = attn[0][0].shape[0]
            last_tok = attn[0][0].shape[2] - 1

            for L in range(n_layer):
                for H in range(n_head):
                    vec = attn[L][0][H, last_tok].to(torch.float32).cpu().numpy()
                    key = f"L{L}_H{H}"
                    (normal if lbl == "normal" else conflict)[key].append(vec)

    scores = score_heads(normal, conflict)
    top10 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    stem = Path(json_path).stem.replace("_instruction", "")
    tgt = Path(out_dir) / f"{stem}_outputs"
    tgt.mkdir(parents=True, exist_ok=True)

    out_json = tgt / "important_heads.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"important_heads": [(k, float(v)) for k, v in top10],
                   "responses": responses}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved → {out_json}")
    print("📌 Top-10 Important Heads:")
    for h, s in top10:
        print(f"  {h:8s} │ {s:8.4f}")

# ========================= 5. CLI Entry =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cuda", type=int, nargs="+", default=[0], help="GPUs to use. Example: --cuda 0 or --cuda 0 1 2")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--lora_path", default="", help="Optional: LoRA adapter path")
    args = parser.parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.cuda])

    if len(args.cuda) == 1:
        idx = args.cuda[0]
        device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
        device_map = {"": 0} if device.type == "cuda" else {"": "cpu"}
    else:
        device = None
        device_map = "auto"

    print(f"🔵 Loading base model from {args.model_path} ...")
    model, tok = load_generic_model(args.model_path, device, device_map)

    if args.lora_path:
        print(f"🟣 Loading LoRA from {args.lora_path} ...")
        model = PeftModel.from_pretrained(model, args.lora_path, device_map=device_map)
        model = model.merge_and_unload()
        print("✅ LoRA merged.")

    detect_heads(args.json_path, model, tok, args.output_dir)

if __name__ == "__main__":
    main()
