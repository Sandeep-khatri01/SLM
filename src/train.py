"""Train a stance keyword model from scratch.

Usage (from repository root):

    python -m src.train --data data/dummy/train.jsonl --out outputs/smoke --max-steps 200
    python -m src.train --data data/pro/train.jsonl --out outputs/pro \
        --n-layer 8 --n-head 6 --n-embd 384 --block-size 256 --max-steps 20000

The training loop is intentionally explicit: no Trainer, no pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from src.data.text import build_vocab, decode, encode, format_record
from src.model.gpt import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny decoder-only LM from scratch")
    p.add_argument("--data", required=True, help="JSONL with 'text' or topic/stance/keywords")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-kv-head", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--max-vocab", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--gen-tokens", type=int, default=16)
    return p.parse_args()


def load_texts(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append(record.get("text") or format_record(record))
    if not texts:
        raise SystemExit(f"no records in {path}")
    return texts


def lr_at(step: int, max_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_type = device.split(":")[0]

    if args.dtype == "auto":
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    ctx = nullcontext() if dtype == torch.float32 else torch.amp.autocast(device_type=device_type, dtype=dtype)

    texts = load_texts(args.data)
    n_val = max(1, int(len(texts) * 0.05)) if len(texts) >= 20 else 0
    train_texts, val_texts = texts[: len(texts) - n_val], texts[len(texts) - n_val :]

    stoi, itos = build_vocab(train_texts, max_size=args.max_vocab)
    train_ids: list[int] = []
    for text in train_texts:
        train_ids.extend(encode(text, stoi))
    val_ids: list[int] = []
    for text in val_texts:
        val_ids.extend(encode(text, stoi))
    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long) if val_ids else train_data

    if len(train_data) <= args.block_size + 1:
        raise SystemExit(
            f"corpus too small: {len(train_data)} tokens, need > {args.block_size + 1}. "
            "Use a smaller --block-size or more data."
        )

    cfg = GPTConfig(
        vocab_size=len(itos),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_kv_head=args.n_kv_head,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"device={device} dtype={dtype} params={n_params/1e6:.2f}M vocab={cfg.vocab_size}")
    print(f"tokens train={len(train_data):,} val={len(val_data):,}")

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == "train" else val_data
        ix = torch.randint(0, len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i : i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + args.block_size] for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps({"itos": itos}, indent=2), encoding="utf-8")

    best_val = float("inf")
    t0 = time.time()
    tokens_seen = 0
    model.train()
    for step in range(args.max_steps):
        lr = lr_at(step, args.max_steps, args.lr, args.min_lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch("train")
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            train_loss += loss.item()
            tokens_seen += x.numel()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 10 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            print(
                f"step {step:5d} | loss {train_loss:.4f} | lr {lr:.2e} | "
                f"{tokens_seen / elapsed:,.0f} tok/s"
            )

        if step > 0 and (step % args.eval_interval == 0 or step == args.max_steps - 1):
            losses = estimate_loss()
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": asdict(cfg)},
                    out_dir / "ckpt.pt",
                )
                print(f"  saved checkpoint (val {best_val:.4f})")

    if not (out_dir / "ckpt.pt").exists():
        torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out_dir / "ckpt.pt")

    prompt = train_texts[0].split("keywords:")[0] + "keywords:"
    ids = encode(prompt, stoi)[:-1]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    sample = model.generate(x, max_new_tokens=args.gen_tokens, temperature=0.8, top_p=0.9)
    print("\n--- sample ---")
    print(decode(sample[0].tolist(), itos))
    print(f"\ndone in {time.time() - t0:.1f}s | best val {best_val:.4f} | saved to {out_dir}")


if __name__ == "__main__":
    main()
