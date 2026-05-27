#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
VERILATOR_ROOT_SYSTEM="/usr/local/share/verilator"
VERILATOR_BIN_SYSTEM="/usr/local/bin/verilator"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Missing Python virtual environment: ${VENV_DIR}"
    echo "Create it first before sourcing this script."
    return 1 2>/dev/null || exit 1
fi

if [ ! -x "${VERILATOR_BIN_SYSTEM}" ]; then
    echo "Missing system Verilator binary: ${VERILATOR_BIN_SYSTEM}"
    echo "Install Verilator globally before sourcing this script."
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "${VERILATOR_ROOT_SYSTEM}/include/verilated.mk" ]; then
    echo "Missing system Verilator root: ${VERILATOR_ROOT_SYSTEM}"
    echo "Expected to find ${VERILATOR_ROOT_SYSTEM}/include/verilated.mk"
    return 1 2>/dev/null || exit 1
fi

export VERILATOR_ROOT="${VERILATOR_ROOT_SYSTEM}"
export PATH="${VENV_DIR}/bin:/usr/local/bin:${PATH}"

echo "Activated tiny-NPU sim environment"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  VERILATOR_ROOT=${VERILATOR_ROOT}"
echo "  python=$(command -v python)"
echo "  cmake=$(command -v cmake)"
echo "  verilator=$(command -v verilator)"
