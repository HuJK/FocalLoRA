#!/bin/bash
set -euo pipefail
export CUDA_VISIABLE_DEVICES="0"
CONDA_BIN=${CONDA_BIN:-}
if [ -z "$CONDA_BIN" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN=$(command -v conda)
  elif [ -x /opt/miniconda/bin/conda ]; then
    CONDA_BIN=/opt/miniconda/bin/conda
  fi
fi

if [ -n "$CONDA_BIN" ]; then
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate focallora
else
  echo "[visualize.sh] Warning: conda not found; running in current environment." >&2
fi

export PROJDIR="/data/local/hujk/FocalLoRA"

MODEL_PATH=${MODEL_PATH:-"$PROJDIR/models/Llama-3.1-8B-Instruct/"}
LORA_PATH=${LORA_PATH:-"$PROJDIR/LoraAdapter/Llama-3.1-8B-Instruct_modified_0.01/batch_0"}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora_path=*)
      LORA_PATH="${1#*=}"
      shift
      ;;
    --lora_path)
      LORA_PATH="$2"
      shift 2
      ;;
    --lora=*)
      LORA_PATH="${1#*=}"
      shift
      ;;
    --lora)
      LORA_PATH="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "[visualize.sh] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

COND1=${COND1:-"Your entire response should be in English, no other language is allowed."}
COND2=${COND2:-"Your entire response should be in French, no other language is allowed."}
TASK=${TASK:-"Describe the greenhouse effect and explain how human activities, such as fossil-fuel combustion, intensify this natural process."}

SYSTEM_PROMPT="$COND1"
NORMAL_USER_PROMPT="$TASK"
CONFLICT_USER_PROMPT="$COND2 $TASK"

JSON_FILE="$PROJDIR/data/visualization_prompts.json"
cat > "$JSON_FILE" <<EOF
[
  {
    "id": "conflict_case",
    "system_message": "$SYSTEM_PROMPT",
    "user_message": "$CONFLICT_USER_PROMPT"
  },
  {
    "id": "normal_case",
    "system_message": "$SYSTEM_PROMPT",
    "user_message": "$NORMAL_USER_PROMPT"
  }
]
EOF

OUTPUT_DIR="$PROJDIR/figure/visualization"
mkdir -p "$OUTPUT_DIR"

derive_model_dir() {
  local raw="${1%/}"
  local base=$(basename "$raw")
  local parent=$(basename "$(dirname "$raw")")
  if [[ "$base" == batch_* ]] || [[ "$base" == checkpoint* ]] || [ "$base" = "adapter" ]; then
    echo "$parent"
  else
    echo "$base"
  fi
}

if [ -n "${LORA_PATH:-}" ]; then
  TARGET_SUBDIR=$(derive_model_dir "$LORA_PATH")
else
  TARGET_SUBDIR=$(derive_model_dir "$MODEL_PATH")
fi

BASE_OUT="$OUTPUT_DIR/$TARGET_SUBDIR"
mkdir -p "$BASE_OUT"

LORA_FLAGS=()
if [ -n "${LORA_PATH:-}" ]; then
  LORA_PREFIX=$(basename "${LORA_PATH%/}")
  LORA_FLAGS+=(--lora_path "$LORA_PATH" --lora_prefix "$LORA_PREFIX")
fi

PYTHON_BIN=${PYTHON_BIN:-python3}

CMD=(
  "$PYTHON_BIN" "$PROJDIR/code/visualization_attention.py"
  --model_path "$MODEL_PATH"
  --json_file "$JSON_FILE"
  --important_file "$PROJDIR/LoraAdapter/Llama-3.1-8B-Instruct/heads.py"
  --output_path "$BASE_OUT"
  --base_prefix "base"
#  --cuda 1
)

if [ "${#LORA_FLAGS[@]}" -gt 0 ]; then
  CMD+=("${LORA_FLAGS[@]}")
fi

"${CMD[@]}"
