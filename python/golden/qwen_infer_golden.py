#!/usr/bin/env python3
"""Golden Qwen tiny inference with GPT-like text prompt handling.

This is a thin driver on top of TinyLLaMAGolden. It keeps the Qwen numeric
path shared with the LLaMA-family golden kernel, and adds a tiny byte-level
text adapter so the current VOCAB_SIZE=256 demo can accept human-readable
prompts and write text outputs.

Important: this is NOT full Qwen tokenizer support. It only uses the first
256 byte-level tokens from Qwen's tokenizer so every token ID stays within
the tiny demo vocabulary.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from golden.llama_infer_golden import TinyLLaMAGolden
from llama_map import (
    HIDDEN,
    N_Q_HEADS,
    N_KV_HEADS,
    HEAD_DIM,
    FFN_DIM,
    N_LAYERS,
    VOCAB_SIZE,
    MAX_SEQ,
    WTE_OFFSET,
    WTE_SIZE,
    BLOCKS_OFFSET,
    LLAMA_BLOCK_SIZE,
    BLK_RMS1_GAMMA,
    BLK_WQ,
    BLK_WK,
    BLK_WV,
    BLK_WO,
    BLK_RMS2_GAMMA,
    BLK_W_GATE,
    BLK_W_UP,
    BLK_W_DOWN,
    BLK_BQ,
    BLK_BK,
    BLK_BV,
    LN_F_OFFSET,
    LN_F_SIZE,
    LM_HEAD_OFFSET,
    LM_HEAD_SIZE,
    WEIGHTS_TOTAL,
    GEMM_SCALE,
    GEMM_SHIFT,
    GEMM_SHIFT_K16,
    GEMM_SHIFT_K128,
)

HF_REPO = "Qwen/Qwen2-0.5B"


def bytes_to_unicode():
    """Reproduce GPT-2 style byte-to-unicode mapping used by Qwen BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class QwenTinyTextCodec:
    """Restricted byte-level text codec backed by Qwen's first 256 tokens."""

    def __init__(self, repo_id=HF_REPO):
        from transformers import AutoTokenizer

        self.repo_id = repo_id
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.byte_encoder = bytes_to_unicode()
        vocab = self.tokenizer.get_vocab()

        self.byte_to_tid = {}
        for byte_val, uchar in self.byte_encoder.items():
            tid = vocab.get(uchar)
            if tid is not None and tid < VOCAB_SIZE:
                self.byte_to_tid[byte_val] = tid

        if len(self.byte_to_tid) != 256:
            raise RuntimeError(
                f"Expected 256 byte-level tiny tokens, found {len(self.byte_to_tid)}"
            )

        self.tid_to_byte = {tid: byte for byte, tid in self.byte_to_tid.items()}

    def encode_text(self, text):
        """Encode UTF-8 text into tiny-Qwen token IDs."""
        prompt_bytes = text.encode("utf-8")
        return [self.byte_to_tid[b] for b in prompt_bytes]

    def decode_tokens(self, token_ids):
        """Decode tiny-Qwen token IDs back to text with best-effort fallback."""
        raw = bytes([self.tid_to_byte.get(t, ord("?")) for t in token_ids])
        return raw.decode("utf-8", errors="replace")


def parse_prompt_ids(text):
    """Parse comma/space separated token IDs."""
    fields = text.replace(",", " ").split()
    if not fields:
        raise ValueError("prompt ID list is empty")
    ids = [int(tok) for tok in fields]
    for tok in ids:
        if tok < 0 or tok >= VOCAB_SIZE:
            raise ValueError(
                f"Prompt token ID {tok} is out of range for VOCAB_SIZE={VOCAB_SIZE}"
            )
    return ids


def load_qwen_demo_weights(weights_bin_path):
    """Load quantized tiny-Qwen weights from weights.bin."""
    with open(weights_bin_path, "rb") as f:
        raw = f.read()
    if len(raw) != WEIGHTS_TOTAL:
        raise ValueError(
            f"Expected {WEIGHTS_TOTAL} bytes in weights.bin, got {len(raw)}"
        )

    buf = np.frombuffer(raw, dtype=np.int8)

    wte = buf[WTE_OFFSET : WTE_OFFSET + WTE_SIZE].reshape(VOCAB_SIZE, HIDDEN).copy()
    blocks_weights = []

    for i in range(N_LAYERS):
        base = BLOCKS_OFFSET + i * LLAMA_BLOCK_SIZE
        blocks_weights.append(
            {
                "rms1_gamma": buf[
                    base + BLK_RMS1_GAMMA : base + BLK_RMS1_GAMMA + HIDDEN
                ].copy(),
                "Wq": buf[
                    base + BLK_WQ : base + BLK_WQ + N_Q_HEADS * HIDDEN * HEAD_DIM
                ]
                .reshape(N_Q_HEADS, HIDDEN, HEAD_DIM)
                .copy(),
                "Wk": buf[
                    base + BLK_WK : base + BLK_WK + N_KV_HEADS * HIDDEN * HEAD_DIM
                ]
                .reshape(N_KV_HEADS, HIDDEN, HEAD_DIM)
                .copy(),
                "Wv": buf[
                    base + BLK_WV : base + BLK_WV + N_KV_HEADS * HIDDEN * HEAD_DIM
                ]
                .reshape(N_KV_HEADS, HIDDEN, HEAD_DIM)
                .copy(),
                "Wo": buf[
                    base + BLK_WO : base + BLK_WO + HIDDEN * HIDDEN
                ].reshape(HIDDEN, HIDDEN).copy(),
                "rms2_gamma": buf[
                    base + BLK_RMS2_GAMMA : base + BLK_RMS2_GAMMA + HIDDEN
                ].copy(),
                "W_gate": buf[
                    base + BLK_W_GATE : base + BLK_W_GATE + HIDDEN * FFN_DIM
                ]
                .reshape(HIDDEN, FFN_DIM)
                .copy(),
                "W_up": buf[
                    base + BLK_W_UP : base + BLK_W_UP + HIDDEN * FFN_DIM
                ]
                .reshape(HIDDEN, FFN_DIM)
                .copy(),
                "W_down": buf[
                    base + BLK_W_DOWN : base + BLK_W_DOWN + FFN_DIM * HIDDEN
                ]
                .reshape(FFN_DIM, HIDDEN)
                .copy(),
                "bq": buf[
                    base + BLK_BQ : base + BLK_BQ + N_Q_HEADS * HEAD_DIM
                ].reshape(N_Q_HEADS, HEAD_DIM).copy(),
                "bk": buf[
                    base + BLK_BK : base + BLK_BK + N_KV_HEADS * HEAD_DIM
                ].reshape(N_KV_HEADS, HEAD_DIM).copy(),
                "bv": buf[
                    base + BLK_BV : base + BLK_BV + N_KV_HEADS * HEAD_DIM
                ].reshape(N_KV_HEADS, HEAD_DIM).copy(),
            }
        )

    ln_f_gamma = buf[LN_F_OFFSET : LN_F_OFFSET + LN_F_SIZE].copy()
    lm_head = buf[LM_HEAD_OFFSET : LM_HEAD_OFFSET + LM_HEAD_SIZE].reshape(
        VOCAB_SIZE, HIDDEN
    ).copy()

    return wte, blocks_weights, ln_f_gamma, lm_head


def generate_tokens(
    model,
    prompt_ids,
    max_new_tokens,
    temperature,
    rng,
    wte,
    blocks_weights,
    ln_f_gamma,
    lm_head,
):
    """Autoregressive decode loop with greedy or temperature sampling."""
    tokens = list(prompt_ids)
    generated = []
    generated_logits = []

    for step in range(max_new_tokens):
        seq_len = len(tokens)
        if seq_len > MAX_SEQ:
            print(f"Warning: seq_len {seq_len} > MAX_SEQ {MAX_SEQ}, stopping")
            break

        logits = model.forward(
            np.array(tokens, dtype=np.int32),
            wte,
            blocks_weights,
            ln_f_gamma,
            lm_head,
        )
        logits_flat = logits[0]

        if temperature > 0.0 and rng is not None:
            logits_f = logits_flat.astype(np.float64)
            logits_f = (logits_f - logits_f.max()) / temperature
            probs = np.exp(logits_f)
            probs = probs / probs.sum()
            next_tok = int(rng.choice(len(probs), p=probs))
        else:
            next_tok = int(np.argmax(logits_flat))

        generated_logits.append(logits_flat.copy())
        generated.append(next_tok)
        tokens.append(next_tok)

        print(
            f"  Step {step}: token={next_tok} "
            f"logits_range=[{logits_flat.min()}, {logits_flat.max()}]"
        )

    return tokens, generated, generated_logits


def main():
    ap = argparse.ArgumentParser(
        description="Run tiny-Qwen golden inference with text prompt handling"
    )
    ap.add_argument("--weights", default=None, help="Path to weights.bin")
    ap.add_argument(
        "--prompt",
        default="Hello",
        help="Input text prompt for tiny byte-level Qwen mode",
    )
    ap.add_argument(
        "--prompt-ids",
        default=None,
        help="Optional comma/space separated token IDs (overrides --prompt)",
    )
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.environ.get("DEMO_OUTDIR", ".")
    weights_path = args.weights or os.path.join(outdir, "weights.bin")
    os.makedirs(outdir, exist_ok=True)

    codec = QwenTinyTextCodec(HF_REPO)
    if args.prompt_ids is not None:
        prompt_ids = parse_prompt_ids(args.prompt_ids)
        prompt_text = codec.decode_tokens(prompt_ids)
        prompt_mode = "ids"
    else:
        prompt_ids = codec.encode_text(args.prompt)
        prompt_text = args.prompt
        prompt_mode = "text"

    if not prompt_ids:
        raise ValueError("Prompt produced no tokens; provide a non-empty prompt")

    print(f"Prompt mode: {prompt_mode}")
    print(f"Prompt text: {prompt_text!r}")
    print(f"Prompt ids: {prompt_ids}")

    wte, blocks_weights, ln_f_gamma, lm_head = load_qwen_demo_weights(weights_path)
    model = TinyLLaMAGolden(
        hidden=HIDDEN,
        n_q_heads=N_Q_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        max_seq=MAX_SEQ,
        scale=GEMM_SCALE,
        shift_k64=GEMM_SHIFT,
        shift_k16=GEMM_SHIFT_K16,
        shift_k128=GEMM_SHIFT_K128,
    )

    rng = np.random.RandomState(args.seed) if args.temperature > 0 else None
    all_tokens, gen_tokens, gen_logits = generate_tokens(
        model,
        prompt_ids,
        args.max_tokens,
        args.temperature,
        rng,
        wte,
        blocks_weights,
        ln_f_gamma,
        lm_head,
    )

    decoded = codec.decode_tokens(all_tokens)
    print(f"\nGolden generated text: {decoded!r}")
    print(f"Golden tokens: {all_tokens}")

    with open(os.path.join(outdir, "prompt_tokens.txt"), "w") as f:
        f.write(" ".join(str(t) for t in prompt_ids) + "\n")

    with open(os.path.join(outdir, "golden_tokens.txt"), "w") as f:
        for t in gen_tokens:
            f.write(f"{t}\n")

    with open(os.path.join(outdir, "golden_text.txt"), "w") as f:
        f.write(decoded + "\n")

    logits_arr = np.array(gen_logits, dtype=np.int8)
    logits_arr.tofile(os.path.join(outdir, "golden_logits.bin"))

    meta = {
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompt": prompt_text if prompt_mode == "text" else None,
        "prompt_ids_arg": args.prompt_ids,
        "prompt_mode": prompt_mode,
        "prompt_ids": prompt_ids,
        "prompt_text": prompt_text,
        "hf_repo": HF_REPO,
        "tokenizer_mode": "tiny_byte_level_first_256",
        "hidden": HIDDEN,
        "n_q_heads": N_Q_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "n_layers": N_LAYERS,
        "ffn_dim": FFN_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_seq": MAX_SEQ,
        "weights": weights_path,
    }
    with open(os.path.join(outdir, "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGolden results saved to {outdir}/")
    print("  prompt_tokens.txt, golden_tokens.txt, golden_text.txt, golden_logits.bin, golden_meta.json")


if __name__ == "__main__":
    main()
