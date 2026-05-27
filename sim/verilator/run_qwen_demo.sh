#!/bin/bash
# =============================================================================
# run_qwen_demo.sh - End-to-end tiny-Qwen inference demo
# Usage:
#   ./run_qwen_demo.sh [--prompt "Hello"] [--prompt-ids "39,68,75,75,78"]
#                      [--max-tokens 10] [--temperature 0.0] [--seed 42]
#                      [--skip-python]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$NPU_DIR/python"
BUILD_DIR="$SCRIPT_DIR/build"
DEMO_OUTDIR="${DEMO_OUTDIR:-$BUILD_DIR/qwen_data_hf}"
VENV_PYTHON="$NPU_DIR/.venv/bin/python"

if [ -x "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

PROMPT="Hello"
PROMPT_IDS=""
MAX_TOKENS=10
SKIP_PYTHON=0
TEMPERATURE=0.0
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt) PROMPT="$2"; shift 2;;
        --prompt-ids) PROMPT_IDS="$2"; shift 2;;
        --max-tokens) MAX_TOKENS="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --skip-python) SKIP_PYTHON=1; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  tiny-Qwen NPU Inference Demo"
if [ -n "$PROMPT_IDS" ]; then
    echo "  prompt_ids: $PROMPT_IDS"
else
    echo "  prompt: '$PROMPT'"
fi
echo "  max_tokens: $MAX_TOKENS"
echo "  temperature: $TEMPERATURE  seed: $SEED"
echo "  outdir: $DEMO_OUTDIR"
echo "============================================"

# Step 1: Python - export Qwen weights and run golden
if [ "$SKIP_PYTHON" -eq 0 ]; then
    echo ""
    echo "--- Step 1: Python weights + golden ---"
    mkdir -p "$DEMO_OUTDIR"

    if ! "$PYTHON_BIN" -c "import transformers, huggingface_hub, safetensors, torch" 2>/dev/null; then
        echo "Installing Python requirements..."
        pip3 install -r "$PYTHON_DIR/requirements.txt"
    fi

    echo "Exporting and packing tiny-Qwen weights..."
    DEMO_OUTDIR="$DEMO_OUTDIR" "$PYTHON_BIN" \
        "$PYTHON_DIR/tools/qwen_gen_weights_hf.py" \
        --outdir "$DEMO_OUTDIR"

    echo "Running tiny-Qwen golden inference..."
    GOLDEN_ARGS=(
        "$PYTHON_DIR/golden/qwen_infer_golden.py"
        --weights "$DEMO_OUTDIR/weights.bin"
        --max-tokens "$MAX_TOKENS"
        --temperature "$TEMPERATURE"
        --seed "$SEED"
        --outdir "$DEMO_OUTDIR"
    )
    if [ -n "$PROMPT_IDS" ]; then
        GOLDEN_ARGS+=(--prompt-ids "$PROMPT_IDS")
    else
        GOLDEN_ARGS+=(--prompt "$PROMPT")
    fi
    DEMO_OUTDIR="$DEMO_OUTDIR" "$PYTHON_BIN" "${GOLDEN_ARGS[@]}"
else
    echo "Skipping Python steps (--skip-python)"
fi

# Step 2: Build llama_demo_infer
echo ""
echo "--- Step 2: Build llama_demo_infer ---"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. 2>&1 | tail -3
cmake --build . --target llama_demo_infer -j"$(nproc)" 2>&1 | tail -5
echo "Build complete."

# Step 3: Run NPU demo
echo ""
echo "--- Step 3: Run NPU inference ---"
./llama_demo_infer --datadir "$DEMO_OUTDIR" --max-tokens "$MAX_TOKENS"

# Step 4: Decode NPU tokens back to text
echo ""
echo "--- Step 4: Decode NPU output ---"
"$PYTHON_BIN" -c "
from python.golden.qwen_infer_golden import QwenTinyTextCodec

codec = QwenTinyTextCodec()

with open('$DEMO_OUTDIR/prompt_tokens.txt') as f:
    prompt_toks = list(map(int, f.read().split()))

with open('$DEMO_OUTDIR/npu_tokens.txt') as f:
    npu_toks = [int(line.strip()) for line in f if line.strip()]

all_toks = prompt_toks + npu_toks
text = codec.decode_tokens(all_toks)
print(f'NPU generated text: \"{text}\"')
print(f'  prompt tokens:    {prompt_toks}')
print(f'  generated tokens: {npu_toks}')
"

echo ""
echo "Demo complete. Output in $DEMO_OUTDIR/"
