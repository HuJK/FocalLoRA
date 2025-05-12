import torch
from transformers import AutoTokenizer
from pathlib import Path

# -----------------------
# Utility functions (same as main script)
# -----------------------

def build_special_ids(tokenizer):
    """Extract special token ids related to system segments."""
    sys_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    eot_id = tokenizer.eos_token_id
    return sys_id, eot_id

def make_sys_mask(input_ids: torch.Tensor, tok) -> torch.Tensor:
    """
    Mark only the system segment based on known chat templates.

    Supported formats:
      1. <|start_header_id|> … <|eot_id|>
      2. ChatML: <|system|> … <|end|>
      3. ChatGPT-im: <|im_start|> system … <|im_end|>
      4. LLaMA/Mistral: [INST] (system)  (user) … [/INST]

    If no format matches, returns all False mask.
    """
    ids = input_ids
    B, L = ids.shape
    mask = torch.zeros_like(ids, dtype=torch.bool)

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
    nl_id = tid("\n")

    for b in range(B):
        row = ids[b].tolist()

        # Format 1: header <|start_header_id|>
        if start_header in row:
            try:
                s = row.index(end_header) + 1
                e = row.index(eot)
                if s < e:
                    mask[b, s:e] = True
                    continue
            except ValueError:
                pass

        # Format 2: <|system|> … <|end|>
        if sys_tok in row:
            try:
                s = row.index(sys_tok) + 1
                e = row.index(end_tok, s)
                mask[b, s:e] = True
                continue
            except ValueError:
                pass

        # Format 3: <|im_start|> system … <|im_end|>
        if im_start in row and im_end in row:
            for pos in [i for i, t in enumerate(row) if t == im_start]:
                if pos + 1 < L and tok.decode([row[pos + 1]]).strip() == "system":
                    s = pos + 2
                    try:
                        e = row.index(im_end, s)
                        mask[b, s:e] = True
                        break
                    except ValueError:
                        pass
            if mask[b].any():
                continue

        # Format 4: [INST] … [/INST]
        if inst_start in row and inst_end in row:
            ist = row.index(inst_start) + 1
            iend = row.index(inst_end)
            split = None
            blank = [(tok.decode([t]).strip() == "") for t in row[ist:iend]]

            for idx in range(len(blank) - 1):
                if blank[idx] and blank[idx + 1]:
                    split = ist + idx
                    break
            if split is None:
                for idx, is_blank in enumerate(blank):
                    if is_blank:
                        split = ist + idx
                        break

            if split is not None and ist < split:
                mask[b, ist:split] = True
            else:
                mask[b, ist:iend] = True

    return mask

def main():
    # [1] Load tokenizer (replace with your own model path)
    model_path = " "
    assert Path(model_path).exists(), f"Model path not found: {model_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    # [2] Sample prompt for debugging
    sample = {
        "system_message": "Please always respond formally and avoid casual expressions.",
        "task": "Define the term 'machine learning'.",
        "user_message": "Make it easy to understand."
    }

    messages = [
        {"role": "system", "content": sample["system_message"]},
        {"role": "user", "content": f"{sample['task']} {sample['user_message']}".strip()}
    ]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # [3] Apply system mask
    sys_mask = make_sys_mask(input_ids, tokenizer)

    # [4] Print tokens with system markers
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
    print("\n===== Token View with System Mask =====")
    for i, (token, is_sys) in enumerate(zip(tokens, sys_mask[0])):
        mark = "🟰" if is_sys else "  "
        print(f"{i:03d} {token.strip():30s} {mark}")
    print("========================================\n")

if __name__ == "__main__":
    main()
