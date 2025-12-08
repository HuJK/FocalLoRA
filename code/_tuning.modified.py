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

import os, json, argparse, math, glob, random, re, pickle
from collections import defaultdict
from typing import List, Tuple, Dict
import random

import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import evallib as evallib

# ---------------- Set random seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Default evaluation dataset (fixed 8-task dev set)
EVAL_DATA_PATH = os.path.join("../data/focal_lora_dataset_dev/dev_eval.json")

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
        tok_inf = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tok_inf = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"
    if tok_inf.pad_token_id is None:
        tok_inf.pad_token = tok.eos_token
        tok_inf.pad_token_id = tok.eos_token_id
        tok_inf.padding_side = "left"
        
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
    return model, tok, tok_inf

# ======================================================
# 3️⃣  Construct system token mask
# ======================================================

def make_sys_mask(input_ids: torch.Tensor, sub_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    input_ids: Tensor (B, N)
    sub_ids:   Tensor (B, M) padded with tokenizer.pad_token_id
    tokenizer: tokenizer object with pad_token_id and decode()
    
    Returns:
        mask: Bool tensor of shape (B, N) with True only for the FIRST match
    """
    pad_id = tokenizer.pad_token_id
    device = input_ids.device

    B, N = input_ids.shape
    _, M = sub_ids.shape

    # Compute true (unpadded) lengths
    sub_lens = (sub_ids != pad_id).sum(dim=1)  # (B,)

    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(B):
        L = sub_lens[b].item()
        if L == 0 or L > N:
            print(f"\n⚠️ Invalid sub length at batch {b}")
            print("input_ids:", tokenizer.decode(input_ids[b], skip_special_tokens=False))
            print("sub_ids:  ", tokenizer.decode(sub_ids[b], skip_special_tokens=False))
            continue

        # Sliding windows
        windows = input_ids[b].unfold(dimension=0, size=L, step=1)  # (N-L+1, L)

        # Target without padding
        target = sub_ids[b, :L]  # (L,)

        full_match = (windows == target).all(dim=1)

        idx = torch.where(full_match)[0]
        if len(idx) > 0:  # ✅ FIRST match only
            start = idx[0].item()
            mask[b, start:start + L] = True
        else:
            # ❌ NOT FOUND → DEBUG OUTPUT
            print(f"\n❌ Subsequence NOT found at batch index {b}")
            print("input_ids:", tokenizer.decode(input_ids[b], skip_special_tokens=False))
            print("sub_ids:  ", tokenizer.decode(sub_ids[b, :L], skip_special_tokens=False))

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

def detect_heads(json_file, model, tok):
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

    scored = sorted(score_heads(nA, cA).items(), key=lambda x: x[1], reverse=True)
    return [(k, float(v)) for k, v in scored]


def save_heads_config(heads, output_dir, model_path, json_path, topk):
    """Cache detected heads so we can resume training without recomputing."""
    os.makedirs(output_dir, exist_ok=True)
    heads_path = os.path.join(output_dir, "heads.json")
    payload = {
        "model_path": model_path,
        "json_path": json_path,
        "topk": str(topk),
        "heads": heads,
    }
    with open(heads_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 Saved heads cache → {heads_path}")


def select_top_heads(all_heads: List[Tuple[str, float]], topk_spec) -> List[Tuple[str, float]]:
    """Select top heads based on numeric count or percentage (e.g., '10p')."""
    if not all_heads:
        return []
    if topk_spec is None:
        return all_heads
    if isinstance(topk_spec, str):
        spec = topk_spec.strip().lower()
    else:
        spec = str(topk_spec)
    if not spec:
        return all_heads

    if spec.endswith("p"):
        try:
            percent = float(spec[:-1])
        except ValueError:
            raise ValueError(f"Invalid percentage for --topk: {topk_spec}")
        count = max(1, math.ceil(percent / 100.0 * len(all_heads)))
    else:
        try:
            count = int(float(spec))
        except ValueError:
            raise ValueError(f"Invalid numeric value for --topk: {topk_spec}")
        count = max(1, count)
    return all_heads[:min(count, len(all_heads))]


def load_heads_config(heads_file: str):
    if not os.path.exists(heads_file):
        raise FileNotFoundError(f"Heads file not found: {heads_file}")
    with open(heads_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_heads = payload.get("heads")
    if raw_heads is None:
        raise ValueError(f"'heads' not defined in {heads_file}")
    heads = [(str(tag), float(score)) for tag, score in raw_heads]
    meta = {
        "model_path": payload.get("model_path"),
        "json_path": payload.get("json_path"),
        "topk": payload.get("topk"),
    }
    return heads, meta


def _extract_suffix_index(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def discover_existing_adapter(out_dir: str):
    if not os.path.isdir(out_dir):
        return None, 0
    candidates = []
    root_config = os.path.join(out_dir, "adapter_config.json")
    if os.path.exists(root_config):
        candidates.append((0, out_dir))
    for entry in os.listdir(out_dir):
        path = os.path.join(out_dir, entry)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            candidates.append((_extract_suffix_index(entry), path))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    resume_path = candidates[-1][1]
    next_idx = candidates[-1][0] + 1 if candidates[-1][0] >= 0 else 0
    return resume_path, next_idx

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
            head_loss += v[m].sum() / v.sum().clamp_min(1e-6) / B
        total_loss += head_loss
        valid_heads += 1

    return 1 - total_loss / max(valid_heads, 1)

# ======================================================
# Dataset and Collate Function for Fine-tuning
# ======================================================

class ConflictDS(Dataset):
    """Dataset for loading conflict samples from multiple JSON files."""

    def __init__(self, json_files: List[str], tokenizer):
        self.samples = []
        self.tokenizer = tokenizer

        for json_file in json_files:
            if not os.path.exists(json_file):
                continue
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Filter for conflict samples only
                conflicts = [s for s in data if s.get('label') == 'conflict']
                self.samples.extend(conflicts)
        random.shuffle(self.samples)
        print(f"📊 Loaded {len(self.samples)} conflict samples from {len(json_files)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate(batch: List[Dict], tokenizer):
    """
    Collate function to batch samples and tokenize them.
    Combines task + user_message as described in the paper.
    """
    conversations = []
    texts_sys = []

    for sample in batch:
        # Combine task and user_message (if present)
        task = sample.get('task', '')
        user_msg = sample.get('user_message', '')

        # Combine as per line 209 logic: task + user_message
        user_content = f"{task} {user_msg}".strip() if user_msg else task

        # Build chat format
        messages = [
            {"role": "system", "content": sample['system_message']},
            {"role": "user", "content": user_content}
        ]
        conversations.append(messages)
        texts_sys.append(sample['system_message'])

    # Apply chat template and tokenize
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]

    # Tokenize with padding
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors='pt'
    )
    encoded_sys = tokenizer(
        texts_sys,
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'],
        'system_ids': encoded_sys['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

# ======================================================
# 6️⃣  Training with LoRA on selected heads
# ======================================================
def save_model(
    model,
    tok,
    out_dir,
    epoch,
    batch_idx,
    current_ratio,
    heads=None,
    eval_data_path: str = EVAL_DATA_PATH,
):
    """Save model/tokenizer and run lightweight eval with detailed logging."""
    print("Running eval and saving model")
    save_dir = os.path.join(out_dir, f"batch_{epoch}_{batch_idx}")
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    # Run quick evaluations
    eval_asr = evallib.quick_eval_asr(
        model,
        tokenizer=tok,
        data_path=eval_data_path,
        heads=heads,
    )
    eval_mmlu = evallib.quick_eval_mmlu(
        model,
        tokenizer=tok
    )

    head_pairs = []
    if heads:
        for tag, _score in heads:
            try:
                l = int(tag.split("_")[0][1:])
                h = int(tag.split("_")[1][1:])
                head_pairs.append((l, h))
            except Exception:
                continue
    # quick_eval_asr handles attention capture internally now; just forward the payload
    detail_payload = {"eval_asr": eval_asr, "eval_mmlu": eval_mmlu}
    with open(os.path.join(save_dir, "detail_log.pkl"), "wb") as f:
        pickle.dump(detail_payload, f)

    # Append training log
    info_file = os.path.join(out_dir, "training_log.csv")
    if not os.path.exists(info_file):
        with open(info_file, "w") as info:
            info.write("epoch,batch_idx,current_ratio,normal_success,conflict_success,both_success,mmlu_acc\n")
    normal_success = eval_asr.get("normal_success") if isinstance(eval_asr, dict) else None
    conflict_success = eval_asr.get("conflict_success") if isinstance(eval_asr, dict) else None
    both_success = eval_asr.get("both_success") if isinstance(eval_asr, dict) else None
    mmlu_acc = eval_mmlu.get("accuracy") if isinstance(eval_mmlu, dict) else None
    with open(info_file, "a") as info:
        info.write(
            f"{epoch},{batch_idx},{current_ratio:.4f},"
            f"{normal_success if normal_success is not None else ''},"
            f"{conflict_success if conflict_success is not None else ''},"
            f"{both_success if both_success is not None else ''},"
            f"{mmlu_acc if mmlu_acc is not None else ''}\n"
        )

    return save_dir, {"asr": eval_asr, "mmlu": eval_mmlu}

def tune(model, tok,tok_inf, heads, data_dir, out_dir, epochs, bs, lr, lam_foc,
         resume_adapter=None, start_batch_idx=0):
    layers = sorted({int(t.split("_")[0][1:]) for t, _ in heads})

    targets = get_lora_targets(model, layers)
    if not targets:
        raise ValueError("No q/k projection layers found!")

    lora_cfg = LoraConfig(r=8, lora_alpha=16, bias="none",
                          target_modules=targets, task_type="CAUSAL_LM")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if resume_adapter:
        if not os.path.exists(resume_adapter):
            raise FileNotFoundError(f"LoRA adapter not found: {resume_adapter}")
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        print(f"♻️ Loaded existing LoRA adapter → {resume_adapter}")
    else:
        model = get_peft_model(model, lora_cfg)

    os.makedirs(out_dir, exist_ok=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total * 100
    print(f"🔧 Trainable parameters: {trainable:,} / {total:,} ({ratio:.4f}% of total)")

    files = glob.glob(os.path.join(data_dir, "*.json"))
    dl = DataLoader(ConflictDS(files, tok), batch_size=bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok))

    opt = torch.optim.AdamW(model.parameters(), lr=lr,)
    total = epochs * math.ceil(len(dl))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)

    model.train()
    current_idx = start_batch_idx
    current_ratio_reached = False
    for ep in range(epochs):
        if current_ratio_reached:
            break
        pbar = tqdm(enumerate(dl), desc=f"Epoch {ep+1}/{epochs}")
        for idx,batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            #breakpoint()
            out = model(**batch, output_attentions=True)
            sys_mask = make_sys_mask(batch["input_ids"], batch["system_ids"],tok)
            loss = lam_foc * focus_loss(out.attentions, sys_mask, heads)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}") 
            current_ratio = float(loss.detach().cpu().item()) / lam_foc
            current_ratio = 1 - current_ratio
            if  idx % 100 == 0:
                save_path, eval_summary = save_model(
                    model, tok_inf, out_dir, current_idx, idx, current_ratio, heads=heads
                )
                eval_metrics = eval_summary.get("asr", {}) if isinstance(eval_summary, dict) else {}
                conflict_success = eval_metrics.get("conflict_success")
                print(f"✅ LoRA adapter checkpoint saved → {save_path} (conflict_success={conflict_success if conflict_success is not None else 'n/a'})")
        save_path, eval_summary = save_model(
            model, tok_inf, out_dir, current_idx, idx, current_ratio, heads=heads
        )
        eval_metrics = eval_summary.get("asr", {}) if isinstance(eval_summary, dict) else {}
        conflict_success = eval_metrics.get("conflict_success")
        print(f"✅ LoRA adapter saved → {save_path} (conflict_success={conflict_success if conflict_success is not None else 'n/a'})")
        current_idx += 1


def main():
    ap = argparse.ArgumentParser("Important-Head LoRA Finetune")
    ap.add_argument("--json_path", required=False, help="Probing file with normal and conflict samples")
    ap.add_argument("--model_path", required=False, help="Base model path")
    ap.add_argument("--tune_path", required=True, help="Folder with conflict samples for fine-tuning")
    ap.add_argument("--output_dir", default="outputs_lora", help="Path to save LoRA adapter")
    ap.add_argument("--lora_path", default=None, help="Optional existing LoRA adapter to load before training")
    ap.add_argument("--topk", type=str, default="10",
                    help="Top-K important heads to select (e.g., 10 or 10p for 10%)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_focus", type=float, default=0.5)
    ap.add_argument("--head_path", type=str, default="", help="Optional path to a precomputed heads.json file.")
    args = ap.parse_args()

    preferred_heads = args.head_path.strip()
    heads_file = preferred_heads or os.path.join(args.output_dir, "heads.json")
    heads_meta = {}
    all_heads = None
    if heads_file and os.path.exists(heads_file):
        all_heads, heads_meta = load_heads_config(heads_file)
        print(f"📂 Loaded cached heads from {heads_file}")
    else:
        if preferred_heads:
            ap.error(f"--head_path specified but not found: {heads_file}")
        if not args.json_path:
            ap.error("--json_path is required when no cached heads are found.")
        if not args.model_path:
            ap.error("--model_path is required when computing new heads.")

    model_path = args.model_path or heads_meta.get("model_path")
    if not model_path:
        ap.error("Base model path missing. Provide --model_path or ensure model_path exists in output_dir/heads.json")

    if args.lora_path:
        if not os.path.exists(args.lora_path):
            ap.error(f"--lora_path not found: {args.lora_path}")
        resume_adapter, start_idx = args.lora_path, 0
        print(f"♻️ Loaded LoRA adapter from --lora_path: {resume_adapter}")
    else:
        resume_adapter, start_idx = discover_existing_adapter(args.output_dir)
        if resume_adapter:
            print(f"♻️ Resuming from existing adapter in output_dir: {resume_adapter}")

    model, tok ,tok_inf= load_model(model_path)

    if all_heads is not None:
        print("📌 Important heads:", all_heads)
    else:
        all_heads = detect_heads(args.json_path, model, tok)
        print("📌 Important heads:", all_heads)
        save_heads_config(all_heads, args.output_dir, model_path, args.json_path, args.topk)

    heads = select_top_heads(all_heads, args.topk)
    print(f"🎯 Using {len(heads)} heads based on topk={args.topk}: {heads}")

    tune(model, tok,tok_inf, heads,
         args.tune_path, args.output_dir,
         args.epochs, args.batch_size, args.lr, args.lambda_focus,
         resume_adapter=resume_adapter, start_batch_idx=start_idx)

if __name__ == "__main__":
    main()
