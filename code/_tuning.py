"""
Focal-Head LoRA Finetune
==========================================

• Selectively fine-tunes "important attention heads" (via LoRA) to enhance LLM alignment with system instructions.
• Key components:
  1) detect_heads  : compares normal vs. conflict attention → selects top-k heads
  2) Q-LoRA (4-bit): injects LoRA only into q/k projection layers with 4-bit quantization
  3) make_sys_mask : builds token-level masks for system segments across chat templates
  4) focus_loss    : encourages final-token attention to return to system region (FP32 for numerical stability)
"""

import os, json, argparse, math, glob, random
from collections import defaultdict
from typing import List, Tuple, Dict

import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- Set random seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================================
# 1️⃣  Locate LoRA target layers (q_proj/k_proj)
# ======================================================

def get_lora_targets(model, layers: List[int]) -> List[str]:
    mtype = (getattr(model.config, "model_type", "") or "").lower()
    archs = [x.lower() for x in getattr(model.config, "architectures", [])]
    if mtype.startswith("qwen2") or any("qwen2" in a for a in archs):
        return [f"model.layers.{i}.self_attn.{p}" for i in layers for p in ("q_proj", "k_proj")]
    if "phi" in mtype or any("phi" in a for a in archs):
        return [f"model.layers.{i}.self_attn.{p}" for i in layers for p in ("q_proj", "k_proj", "qkv_proj")]
    if mtype in {"llama", "mistral"} or "llama" in mtype:
        return [f"model.layers.{i}.self_attn.{p}" for i in layers for p in ("q_proj", "k_proj")]
    # fallback for unknown models
    cand = []
    for name, _ in model.named_modules():
        if any(f".{i}." in name for i in layers) and name.split(".")[-1] in {
            "q_proj", "k_proj", "qkv_proj", "c_attn", "query_key_value"}:
            cand.append(name)
    return cand

# ======================================================
# 2️⃣  Load model with 4-bit quantization
# ======================================================

def load_model(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

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
        attn_implementation="eager",
    )
    return model, tok

# ======================================================
# 3️⃣  Construct system token mask
# ======================================================

def make_sys_mask(input_ids: torch.Tensor, tok) -> torch.Tensor:
    B, L = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    tid = tok.convert_tokens_to_ids
    start_header = tid("<|start_header_id|>")
    end_header = tid("<|end_header_id|>")
    eot = tok.eos_token_id
    sys_tok = tid("<|system|>")
    end_tok = tid("<|end|>")
    im_start = tid("<|im_start|>")
    im_end = tid("<|im_end|>")
    inst_start = tid("[INST]")
    inst_end = tid("[/INST]")

    for b in range(B):
        row = input_ids[b].tolist()
        # Format a: header template
        if start_header in row:
            try:
                s = row.index(end_header) + 1
                e = row.index(eot)
                mask[b, s:e] = True
                continue
            except ValueError:
                pass
        # Format b: ChatML <|system|>
        if sys_tok in row:
            try:
                s = row.index(sys_tok) + 1
                e = row.index(end_tok, s)
                mask[b, s:e] = True
                continue
            except ValueError:
                pass
        # Format c: OpenChat <|im_start|> system <|im_end|>
        if im_start in row and im_end in row:
            for pos in [i for i, t in enumerate(row) if t == im_start]:
                if pos + 1 < L and tok.decode([row[pos + 1]]).strip() == "system":
                    s = pos + 2
                    e = row.index(im_end, s)
                    mask[b, s:e] = True
                    break
            if mask[b].any():
                continue
        # Format d: [INST]...[/INST]
        if inst_start in row and inst_end in row:
            ist = row.index(inst_start) + 1
            iend = row.index(inst_end)
            split = None
            for i in range(ist, iend - 1):
                if input_ids[b, i].item() == eot and input_ids[b, i + 1].item() == eot:
                    split = i
                    break
            if split is None:
                for i in range(ist, iend):
                    if tok.decode([row[i]]).isspace():
                        split = i
                        break
            if split and ist < split:
                mask[b, ist:split] = True
            else:
                mask[b, ist:iend] = True
    return mask

# ======================================================
# 4️⃣  Identify important attention heads
# ======================================================

def trim_and_stack(rows):
    m = min(len(r) for r in rows)
    return np.stack([r[:m] for r in rows])

def trim_same(a, b):
    m = min(a.shape[1], b.shape[1])
    return a[:, :m], b[:, :m]

def score_heads(norm, conf):
    scores = {}
    for k in norm:
        if k not in conf:
            continue
        try:
            n = trim_and_stack(norm[k])
            c = trim_and_stack(conf[k])
            n, c = trim_same(n, c)
        except Exception:
            continue
        p, q = [np.exp(x - np.max(x, -1, keepdims=True)) for x in (n, c)]
        p /= p.sum(-1, keepdims=True)
        q /= q.sum(-1, keepdims=True)
        kl = (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum() / p.shape[0]
        shift = np.mean(np.abs(n.mean(1) - c.mean(1)))
        frob = np.linalg.norm(n - c, ord="fro")
        scores[k] = 0.4 * frob + 0.3 * shift + 0.3 * kl
    return scores

def extract_attn(model, tok, sys_msg, usr_msg):
    text = tok.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user",   "content": usr_msg}],
        tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_attentions=True)
    return out.attentions

def detect_heads(json_file, model, tok, k=10):
    data = json.load(open(json_file, encoding="utf-8"))
    grp = defaultdict(lambda: {"normal": None, "conflict": None})
    for s in data:
        bid = s["id"].replace("_normal", "").replace("_conflict", "")
        grp[bid][s["label"]] = s

    nA, cA = defaultdict(list), defaultdict(list)
    for pair in tqdm(grp.values(), desc="Extract"):
        for lab in ("normal", "conflict"):
            if pair[lab] is None:
                continue
            s = pair[lab]
            usr = f"{s['task']} {s['user_message']}".strip() or s["task"]
            attn = extract_attn(model, tok, s["system_message"], usr)
            last = attn[0][0].shape[2] - 1
            for l in range(len(attn)):
                for h in range(attn[l][0].shape[0]):
                    row = attn[l][0][h, last, :].float().cpu().numpy()
                    (nA if lab == "normal" else cA)[f"L{l}_H{h}"].append(row)

    imp = sorted(score_heads(nA, cA).items(), key=lambda x: x[1], reverse=True)[:k]
    return [(k, float(v)) for k, v in imp]

# ======================================================
# 5️⃣  Focus Loss: encourages attention to system region
# ======================================================

def focus_loss(attns, sys_mask, heads):
    B = sys_mask.size(0)
    total_loss = torch.zeros([], dtype=torch.float32, device=sys_mask.device)
    valid_heads = 0

    for tag, _ in heads:
        l = int(tag.split("_")[0][1:])
        h = int(tag.split("_H")[1])
        A = attns[l][:, h].float()
        last = A.size(1) - 1
        head_loss = torch.zeros([], dtype=torch.float32, device=sys_mask.device)
        for b in range(B):
            m = sys_mask[b]
            if not m.any():
                continue
            v = A[b, last]
            head_loss -= v[m].sum() / v.sum().clamp_min(1e-6)
        total_loss += head_loss
        valid_heads += 1

    return total_loss / max(valid_heads, 1)

# ======================================================
# 6️⃣  Training with LoRA on selected heads
# ======================================================

def tune(model, tok, heads, data_dir, out_dir, epochs, bs, lr, lam_foc):
    layers = sorted({int(t.split("_")[0][1:]) for t, _ in heads})
    if not targets:
        raise ValueError("No q/k projection layers found!")

    lora_cfg = LoraConfig(r=8, lora_alpha=16, bias="none",
                          target_modules=targets, task_type="CAUSAL_LM")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total * 100
    print(f"🔧 Trainable parameters: {trainable:,} / {total:,} ({ratio:.4f}% of total)")

    files = glob.glob(os.path.join(data_dir, "*.json"))
    dl = DataLoader(ConflictDS(files, tok), batch_size=bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total = epochs * math.ceil(len(dl))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)

    model.train()
    for ep in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch, output_attentions=True)
            sys_mask = make_sys_mask(batch["input_ids"], tok)
            loss = lam_foc * focus_loss(out.attentions, sys_mask, heads)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"✅ LoRA adapter saved → {out_dir}")


def main():
    ap = argparse.ArgumentParser("Important-Head LoRA Finetune")
    ap.add_argument("--json_path", required=True, help="Probing file with normal and conflict samples")
    ap.add_argument("--model_path", required=True, help="Base model path")
    ap.add_argument("--tune_path", required=True, help="Folder with conflict samples for fine-tuning")
    ap.add_argument("--output_dir", default="outputs_lora", help="Path to save LoRA adapter")
    ap.add_argument("--topk", type=int, default=10, help="Top-K important heads to select")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_focus", type=float, default=0.5)
    args = ap.parse_args()

    model, tok = load_model(args.model_path)
    heads = detect_heads(args.json_path, model, tok, k=args.topk)
    print("📌 Important heads:", heads)

    tune(model, tok, heads,
         args.tune_path, args.output_dir,
         args.epochs, args.batch_size, args.lr, args.lambda_focus)

if __name__ == "__main__":
    main()
