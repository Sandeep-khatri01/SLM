"""Sample keywords from a trained stance model.

Run from repository root:

    python -m src.generate --ckpt outputs/pro/ckpt.pt --topic "AI regulation"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.text import decode, encode
from src.model.gpt import GPT, GPTConfig


def load(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocab = json.loads((Path(ckpt_path).parent / "vocab.json").read_text(encoding="utf-8"))
    return model, vocab["itos"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--stance", default="positive")
    p.add_argument("--tokens", type=int, default=18)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--samples", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, itos = load(args.ckpt, device)
    stoi = {w: i for i, w in enumerate(itos)}
    prompt = f"topic: {args.topic.lower()} | stance: {args.stance} | keywords:"
    ids = encode(prompt, stoi)[:-1]
    x = torch.tensor([ids], dtype=torch.long, device=device).repeat(args.samples, 1)
    out = model.generate(x, max_new_tokens=args.tokens, temperature=args.temperature, top_p=0.9)
    for row in out:
        print(decode(row.tolist(), itos))


if __name__ == "__main__":
    main()
