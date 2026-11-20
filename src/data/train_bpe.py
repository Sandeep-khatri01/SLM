"""Train a Byte-Pair Encoding (BPE) tokenizer from scratch for T1 scaling.

This tokenizer operates on raw bytes (ByteLevel BPE), meaning:
1. Zero out-of-vocabulary tokens (any UTF-8 string can be encoded)
2. Preserves whitespace and punctuation without external rules
3. No reliance on pretrained checkpoints or external APIs
4. Scales smoothly from 1.4M to 10B+ tokens

Usage:
    python -m src.data.train_bpe --inputs data/curated/topics.jsonl --out outputs/bpe-32k --vocab-size 1000
    python -m src.data.train_bpe --inputs corpus/*.txt --out outputs/bpe-32k --vocab-size 32000
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer from scratch")
    p.add_argument("--inputs", nargs="+", required=True, help="Input text or jsonl files/patterns")
    p.add_argument("--out", required=True, help="Output directory to save tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    p.add_argument("--min-frequency", type=int, default=2, help="Min token frequency")
    return p.parse_args()


def text_iterator(input_paths: list[str]) -> Iterator[str]:
    """Yield text lines from .txt or .jsonl files."""
    for pattern in input_paths:
        matched_files = glob.glob(pattern) or [pattern]
        for fpath in matched_files:
            path = Path(fpath)
            if not path.is_file():
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if path.suffix == ".jsonl":
                        try:
                            record = json.loads(line)
                            # Yield formatted prompt string if topic record
                            if "topic" in record:
                                pro_kw = ", ".join(record.get("pro", []))
                                con_kw = ", ".join(record.get("con", []))
                                yield f"topic: {record['topic']} | domain: {record.get('domain', '')}"
                                if pro_kw:
                                    yield f"topic: {record['topic']} | stance: positive | keywords: {pro_kw}"
                                if con_kw:
                                    yield f"topic: {record['topic']} | stance: negative | keywords: {con_kw}"
                            elif "text" in record:
                                yield record["text"]
                        except json.JSONDecodeError:
                            yield line
                    else:
                        yield line


def train_bpe(
    inputs: list[str],
    out_dir: Path | str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> Tokenizer:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize BPE model with ByteLevel pre-tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab={vocab_size:,}) from inputs: {inputs}...")
    tokenizer.train_from_iterator(text_iterator(inputs), trainer=trainer)

    save_file = out_path / "tokenizer.json"
    tokenizer.save(str(save_file))
    print(f"✓ Saved tokenizer to {save_file} (actual vocab size: {tokenizer.get_vocab_size():,})")
    return tokenizer


def main() -> None:
    args = parse_args()
    train_bpe(
        inputs=args.inputs,
        out_dir=args.out,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
