#!/bin/bash
set -euo pipefail

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
  echo "[test_model.sh] Warning: conda not found; running in current environment." >&2
fi

export PROJDIR="/data/local/hujk/FocalLoRA"

MODEL_PATH=${MODEL_PATH:-"$PROJDIR/models/Llama-3.1-8B-Instruct/"}
DEFAULT_LORA="$PROJDIR/LoraAdapter/Llama-3.1-8B-Instruct_modified_0.01/batch_0"
LORA_PATH=${LORA_PATH:-"$DEFAULT_LORA"}

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
      echo "[test_model.sh] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

export COND1=${COND1:-"To make your response easy to scan, please organize your answer as exactly five numbered items (1.-5.). Use clear list formatting with each point on a separate line."}
export COND2=${COND2:-"I prefer reading continuous text rather than bullet points. Please provide one single paragraph with no list formatting or line breaks. Make it flow naturally as prose."}
export TASK=${TASK:-"Describe the greenhouse effect and explain how human activities, such as fossil-fuel combustion, intensify this natural process."}

PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-0}

CMD=(
  "$PYTHON_BIN" "$PROJDIR/code/_testmodel.py"
  --model_path "$MODEL_PATH"
  --cuda_device "$CUDA_DEVICE"
  --max_new_tokens "${MAX_NEW_TOKENS:-512}"
)

if [ -n "${LORA_PATH:-}" ]; then
  CMD+=(--lora_path "$LORA_PATH")
else
  CMD+=(--lora_path "")
fi

"${CMD[@]}"
