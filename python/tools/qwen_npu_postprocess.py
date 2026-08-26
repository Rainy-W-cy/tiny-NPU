#!/usr/bin/env python3
"""Decode tiny-Qwen NPU token IDs and write npu_text.txt."""

import argparse
import os
import sys


PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from golden.qwen_infer_golden import QwenTinyTextCodec, VOCAB_SIZE


def read_token_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        fields = f.read().split()
    try:
        token_ids = [int(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"Invalid token ID in {path}") from exc
    for token_id in token_ids:
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise ValueError(
                f"Token ID {token_id} in {path} is outside [0, {VOCAB_SIZE - 1}]"
            )
    return token_ids


def main():
    parser = argparse.ArgumentParser(
        description="Convert tiny-Qwen NPU token IDs into UTF-8 text"
    )
    parser.add_argument(
        "--datadir",
        default=None,
        help="Directory containing prompt_tokens.txt and npu_tokens.txt",
    )
    parser.add_argument("--prompt-tokens", default=None)
    parser.add_argument("--npu-tokens", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    datadir = args.datadir or os.environ.get("DEMO_OUTDIR", ".")
    prompt_path = args.prompt_tokens or os.path.join(datadir, "prompt_tokens.txt")
    npu_path = args.npu_tokens or os.path.join(datadir, "npu_tokens.txt")
    output_path = args.output or os.path.join(datadir, "npu_text.txt")

    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt token file not found: {prompt_path}")
    if not os.path.isfile(npu_path):
        raise FileNotFoundError(f"NPU token file not found: {npu_path}")

    prompt_tokens = read_token_ids(prompt_path)
    npu_tokens = read_token_ids(npu_path)
    if not prompt_tokens:
        raise ValueError(f"Prompt token file is empty: {prompt_path}")

    codec = QwenTinyTextCodec()
    npu_text = codec.decode_tokens(prompt_tokens + npu_tokens)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(npu_text + "\n")

    print(f"NPU generated text : {npu_text!r}")
    print(f"Prompt token IDs   : {prompt_tokens}")
    print(f"NPU token IDs      : {npu_tokens}")
    print(f"NPU text saved to  : {output_path}")


if __name__ == "__main__":
    main()
