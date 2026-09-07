"""Activation extraction from HuggingFace weights. NO AZURE DEPENDENCY.

This is the boundary of the project. Everything upstream needed hosted inference and is
now frozen on disk. Everything from here runs on open weights that can be downloaded
indefinitely, so the research cannot be halted by loss of access.

Model choice
------------
Phi-4-mini-instruct (3.8B) is the PRIMARY target, not the fallback. In fp16 it is ~7.6GB
and fits a single free T4 on Kaggle or Colab, so the geometry work requires no paid
compute at all. Phi-4 (14B, ~28GB) is the secondary target for 2xT4 or a CPU box with
enough RAM.

This inversion matters: it takes compute off the critical path entirely.

What is extracted
-----------------
A single PROMPT-SIDE forward pass per item with output_hidden_states=True. No generation.
That is roughly 300x cheaper than sampling and is all the geometry needs, because the
research question is whether the outcome is predictable BEFORE the model responds.

Two views are cached per layer:
  pooled     mean over non-padding tokens          -> LID, linear probes
  per_token  the last N token states, fp16         -> curvature along the trajectory

Curvature needs the token sequence; LID does not. Caching both means the geometry stage
never has to touch a GPU again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen"
ACTS = ROOT / "data" / "activations"

MODELS = {
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4": "microsoft/phi-4",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}

# Which frozen corpus supplies the prompts, and how to reconstruct the model-visible text.
CORPORA = {
    "safety_single": ("corpus_labelled.parquet", ["prompt"]),
    "safety_mt": ("corpus_mt_labelled.parquet", ["turn1_user", "turn2_assistant", "turn3_user"]),
    "safety_exp": ("corpus_exp_labelled.parquet", ["prompt"]),
    "factual": ("corpus_fc_labelled.parquet", ["turn1_user", "turn2_assistant", "turn3_user"]),
}


def build_chat(row: dict, cols: list[str], tokenizer) -> str:
    """Rebuild the exact conversation the target model saw, via its own chat template.

    Using the template rather than raw concatenation matters: special tokens and role
    markers are part of the input geometry, and omitting them would measure a different
    distribution from the one that produced the labels.
    """
    if len(cols) == 1:
        msgs = [{"role": "user", "content": str(row[cols[0]])}]
    else:
        msgs = [
            {"role": "user", "content": str(row[cols[0]])},
            {"role": "assistant", "content": str(row[cols[1]])},
            {"role": "user", "content": str(row[cols[2]])},
        ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi4-mini", choices=list(MODELS))
    ap.add_argument("--corpus", default="factual", choices=list(CORPORA))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--keep-tokens", type=int, default=64,
                    help="trailing token states retained per layer for curvature")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="keep every Nth layer for the per-token cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, cols = CORPORA[args.corpus]
    df = pd.read_parquet(FROZEN / src)
    if args.limit:
        df = df.head(args.limit)
    print(f"[extract] {len(df)} rows from {src}")

    repo = MODELS[args.model]
    dtype = getattr(torch, args.dtype)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32  # fp16 matmul is not supported on most CPU backends
    print(f"[extract] {repo} on {device} ({dtype})")

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, trust_remote_code=True,
        device_map=device if device == "cuda" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for the embedding output
    keep_layers = list(range(0, n_layers, args.layer_stride))
    print(f"[extract] {n_layers} layers, caching {len(keep_layers)}")

    texts = [build_chat(r._asdict(), cols, tok) for r in df.itertuples()]
    rids = df["rid"].tolist()

    pooled = np.zeros((len(df), n_layers, model.config.hidden_size), dtype=np.float16)
    per_tok = np.zeros((len(df), len(keep_layers), args.keep_tokens, model.config.hidden_size),
                       dtype=np.float16)
    lengths = np.zeros(len(df), dtype=np.int32)

    with torch.inference_mode():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i : i + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length, add_special_tokens=False).to(device)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1)

            for li, h in enumerate(out.hidden_states):
                # Mean over real tokens only; padding would drag the mean toward zero and
                # make pooled geometry a function of batch composition.
                summed = (h * mask).sum(dim=1)
                pooled[i : i + len(chunk), li] = (
                    (summed / mask.sum(dim=1).clamp(min=1)).float().cpu().numpy().astype(np.float16)
                )

            for ki, li in enumerate(keep_layers):
                h = out.hidden_states[li]
                for bi in range(len(chunk)):
                    L = int(enc["attention_mask"][bi].sum())
                    s = max(0, L - args.keep_tokens)
                    seg = h[bi, s:L].float().cpu().numpy().astype(np.float16)
                    per_tok[i + bi, ki, : seg.shape[0]] = seg

            lengths[i : i + len(chunk)] = enc["attention_mask"].sum(dim=1).cpu().numpy()

            if (i // args.batch_size) % 10 == 0:
                print(f"[extract] {min(i + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    ACTS.mkdir(parents=True, exist_ok=True)
    out_path = ACTS / f"{args.model}__{args.corpus}.npz"
    np.savez_compressed(
        out_path,
        rids=np.array(rids), pooled=pooled, per_token=per_tok,
        lengths=lengths, keep_layers=np.array(keep_layers),
    )
    meta = {
        "model": repo, "corpus": args.corpus, "n_items": len(df),
        "n_layers": n_layers, "keep_layers": keep_layers,
        "hidden_size": model.config.hidden_size,
        "keep_tokens": args.keep_tokens, "max_length": args.max_length,
        "dtype": str(dtype), "device": device,
    }
    (ACTS / f"{args.model}__{args.corpus}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[extract] -> {out_path}  pooled={pooled.shape}  per_token={per_tok.shape}")


if __name__ == "__main__":
    main()
