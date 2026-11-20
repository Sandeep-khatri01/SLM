"""Measure training throughput for the real ~20M model config.

Run from repository root:

    python scripts/bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.gpt import GPT, GPTConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = "cuda"
    cfg = GPTConfig(vocab_size=16384, block_size=256, n_layer=8, n_head=6, n_embd=384)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    batch = 64
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    for _ in range(3):
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    steps = 10
    for _ in range(steps):
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens = steps * x.numel()
    print(f"params      {model.num_params()/1e6:.1f}M")
    print(f"throughput  {tokens/dt:,.0f} tok/s  ({tokens/dt*3600/1e6:.1f}M tok/hour)")
    print(f"peak VRAM   {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"step time   {dt/steps*1000:.0f} ms for {batch}x{cfg.block_size} tokens")


if __name__ == "__main__":
    main()
