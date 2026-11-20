"""Tiny word-level tokenizer and record formatting.

A word-level vocabulary is deliberate: the target outputs are keyword lists, so
a 10k-20k word vocab is plenty, and it removes a dependency on any tokenizer
library or pretrained tokenizer.
"""

from __future__ import annotations

import re
from collections import Counter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[,.:;!?|]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(
    texts: list[str], max_size: int = 20000, min_freq: int = 1
) -> tuple[dict[str, int], list[str]]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    words = [w for w, c in counts.most_common() if c >= min_freq]
    words = words[: max_size - len(SPECIALS)]
    itos = SPECIALS + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int], add_bos: bool = True) -> list[int]:
    ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokenize(text)]
    return ([stoi["<bos>"]] if add_bos else []) + ids + [stoi["<eos>"]]


def decode(ids: list[int], itos: list[str]) -> str:
    special = set(SPECIALS)
    out = []
    for i in ids:
        if i < 0 or i >= len(itos):
            continue
        tok = itos[i]
        if tok in special:
            continue
        out.append(tok)
    return " ".join(out)


def format_record(record: dict) -> str:
    """Canonical training string for a (topic, stance, keywords) record."""
    keywords = record["keywords"]
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    kw = ", ".join(keywords)
    return (
        f"topic: {record['topic'].strip().lower()} | "
        f"stance: {record['stance'].strip().lower()} | "
        f"keywords: {kw.lower()}"
    )
