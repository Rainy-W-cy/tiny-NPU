#!/bin/bash
# =============================================================================
# run_qwen_demo.sh - Staged tiny-Qwen inference demo
#
# Default flow:
#   Reuse weights.bin when present, then run golden, build and infer.
#
# Stage selection:
#   --stages weights,golden,build,infer
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$NPU_DIR/python"
BUILD_DIR="$SCRIPT_DIR/build"
DEMO_OUTDIR="${DEMO_OUTDIR:-$BUILD_DIR/qwen_data_hf}"

VENV_PYTHON="$NPU_DIR/.venv/bin/python"
VENV_CMAKE="$NPU_DIR/.venv/bin/cmake"

if [ -x "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

if [ -x "$VENV_CMAKE" ]; then
    CMAKE_BIN="$VENV_CMAKE"
else
    CMAKE_BIN="cmake"
fi

PROMPT="Hello"
PROMPT_IDS=""
PROMPT_EXPLICIT=0
MAX_TOKENS=10
TEMPERATURE=0.0
SEED=42
KV_CACHE=0
STAGES=""
FORCE_WEIGHTS=0
SKIP_PYTHON=0

usage() {
    cat <<'EOF'
Usage: run_qwen_demo.sh [options]

Inference options:
  --prompt TEXT            Text prompt for the golden/prompt stage
  --prompt-ids IDS         Comma/space separated prompt token IDs
  --max-tokens N           Maximum generated token count
  --temperature VALUE      Python golden sampling temperature
  --seed N                 Python golden random seed
  --kv-cache               Use the Prefill + Decode NPU path

Stage options:
  --stages LIST            Comma-separated stages: weights,golden,build,infer
  --force-weights          Regenerate weights in the default flow
  --skip-python            Legacy: skip weights and golden stages
  -h, --help               Show this help

Without --stages, existing weights.bin is reused automatically and the script
runs golden, build and infer. The infer stage also writes npu_text.txt.
EOF
}

require_value() {
    if [ "$#" -lt 2 ]; then
        echo "ERROR: $1 requires a value" >&2
        exit 1
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prompt)
            require_value "$@"
            PROMPT="$2"
            PROMPT_EXPLICIT=1
            shift 2
            ;;
        --prompt-ids)
            require_value "$@"
            PROMPT_IDS="$2"
            PROMPT_EXPLICIT=1
            shift 2
            ;;
        --max-tokens)
            require_value "$@"
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            require_value "$@"
            TEMPERATURE="$2"
            shift 2
            ;;
        --seed)
            require_value "$@"
            SEED="$2"
            shift 2
            ;;
        --stages)
            require_value "$@"
            STAGES="$2"
            shift 2
            ;;
        --kv-cache)
            KV_CACHE=1
            shift
            ;;
        --force-weights)
            FORCE_WEIGHTS=1
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

DO_WEIGHTS=0
DO_GOLDEN=0
DO_BUILD=0
DO_INFER=0

if [ -z "$STAGES" ]; then
    DO_GOLDEN=1
    DO_BUILD=1
    DO_INFER=1
    if [ ! -s "$DEMO_OUTDIR/weights.bin" ]; then
        DO_WEIGHTS=1
    fi
else
    IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
    for stage in "${REQUESTED_STAGES[@]}"; do
        case "$stage" in
            weights) DO_WEIGHTS=1 ;;
            golden)  DO_GOLDEN=1 ;;
            build)   DO_BUILD=1 ;;
            infer)   DO_INFER=1 ;;
            all)
                DO_WEIGHTS=1
                DO_GOLDEN=1
                DO_BUILD=1
                DO_INFER=1
                ;;
            "") ;;
            *)
                echo "ERROR: Unknown stage '$stage'" >&2
                echo "Valid stages: weights,golden,build,infer" >&2
                exit 1
                ;;
        esac
    done
fi

if [ "$FORCE_WEIGHTS" -eq 1 ]; then
    DO_WEIGHTS=1
fi

if [ "$SKIP_PYTHON" -eq 1 ]; then
    DO_WEIGHTS=0
    DO_GOLDEN=0
fi

mkdir -p "$DEMO_OUTDIR"
LOG_FILE="${LOG_FILE:-$DEMO_OUTDIR/qwen_run_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

selected_stages=()
[ "$DO_WEIGHTS" -eq 1 ] && selected_stages+=(weights)
[ "$DO_GOLDEN" -eq 1 ] && selected_stages+=(golden)
[ "$DO_BUILD" -eq 1 ] && selected_stages+=(build)
[ "$DO_INFER" -eq 1 ] && selected_stages+=(infer)

echo "Log file: $LOG_FILE"
echo "============================================"
echo "  tiny-Qwen NPU Inference Demo"
if [ -n "$PROMPT_IDS" ]; then
    echo "  prompt_ids: $PROMPT_IDS"
else
    echo "  prompt: '$PROMPT'"
fi
echo "  max_tokens: $MAX_TOKENS"
echo "  temperature: $TEMPERATURE  seed: $SEED"
echo "  kv_cache: $KV_CACHE"
echo "  stages: ${selected_stages[*]:-(none)}"
echo "  outdir: $DEMO_OUTDIR"
echo "============================================"

ensure_python_requirements() {
    if ! "$PYTHON_BIN" -c \
        "import transformers, huggingface_hub, safetensors, torch, numpy" \
        2>/dev/null; then
        echo "Installing Python requirements..."
        "$PYTHON_BIN" -m pip install -r "$PYTHON_DIR/requirements.txt"
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1" >&2
        echo "       Run the required earlier stage or include it in --stages." >&2
        exit 1
    fi
}

if [ "$DO_WEIGHTS" -eq 1 ]; then
    echo ""
    echo "--- Stage: weights ---"
    ensure_python_requirements
    echo "Exporting and packing tiny-Qwen weights..."
    DEMO_OUTDIR="$DEMO_OUTDIR" "$PYTHON_BIN" \
        "$PYTHON_DIR/tools/qwen_gen_weights_hf.py" \
        --outdir "$DEMO_OUTDIR"
elif [ -s "$DEMO_OUTDIR/weights.bin" ] && \
     { [ "$DO_GOLDEN" -eq 1 ] || [ "$DO_INFER" -eq 1 ]; }; then
    echo ""
    echo "--- Stage: weights (reused) ---"
    echo "Using existing weights: $DEMO_OUTDIR/weights.bin"
fi

if [ "$DO_GOLDEN" -eq 1 ]; then
    echo ""
    echo "--- Stage: golden ---"
    require_file "$DEMO_OUTDIR/weights.bin"
    ensure_python_requirements

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
elif [ "$DO_INFER" -eq 1 ] && [ "$PROMPT_EXPLICIT" -eq 1 ]; then
    echo "WARNING: golden stage is disabled; the requested prompt will not update"
    echo "         prompt_tokens.txt. Infer will use the existing token file."
fi

if [ "$DO_BUILD" -eq 1 ]; then
    echo ""
    echo "--- Stage: build ---"
    mkdir -p "$BUILD_DIR"
    "$CMAKE_BIN" -S "$SCRIPT_DIR" -B "$BUILD_DIR"
    "$CMAKE_BIN" --build "$BUILD_DIR" --target qwen_demo_infer -j"$(nproc)"
    echo "Build complete."
fi

if [ "$DO_INFER" -eq 1 ]; then
    echo ""
    echo "--- Stage: infer / NPU execution ---"
    require_file "$DEMO_OUTDIR/weights.bin"
    require_file "$DEMO_OUTDIR/prompt_tokens.txt"
    require_file "$BUILD_DIR/qwen_demo_infer"
    ensure_python_requirements

    RUN_ARGS=(
        "$BUILD_DIR/qwen_demo_infer"
        --datadir "$DEMO_OUTDIR"
        --max-tokens "$MAX_TOKENS"
    )
    if [ "$KV_CACHE" -eq 1 ]; then
        RUN_ARGS+=(--kv-cache)
    fi
    "${RUN_ARGS[@]}"

    echo ""
    echo "--- Stage: infer / NPU text postprocess ---"
    "$PYTHON_BIN" "$PYTHON_DIR/tools/qwen_npu_postprocess.py" \
        --datadir "$DEMO_OUTDIR"
fi

echo ""
echo "Demo stages complete. Output in $DEMO_OUTDIR/"
