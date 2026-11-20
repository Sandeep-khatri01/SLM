"""Unsupervised keyword extraction (RAKE-style, stdlib only).

No neural model, no pretrained weights: candidates are scored by word
co-occurrence and phrase length, which is exactly the "statistical supervision"
the project needs.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z][a-z'\-]{2,}")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")

STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down during
    each few for from further had has have having he her here hers herself him himself his
    how i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves may might must shall upon within without many
    much often however therefore thus since although though among across per via etc
    """.split()
)


def _phrases(text: str) -> list[list[str]]:
    phrases: list[list[str]] = []
    for sentence in _SENT.split(text):
        current: list[str] = []
        for word in _WORD.findall(sentence.lower()):
            if word in STOPWORDS:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(word)
        if current:
            phrases.append(current)
    return phrases


def extract_keywords(text: str, top_k: int = 25, max_len: int = 3) -> list[tuple[str, float]]:
    """Return [(phrase, score)] ranked by RAKE score."""
    phrases = _phrases(text)
    freq: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in phrases:
        for word in phrase:
            freq[word] += 1
            degree[word] += len(phrase)
    word_score = {w: degree[w] / freq[w] for w in freq}

    scored: dict[str, float] = {}
    for phrase in phrases:
        if not (1 <= len(phrase) <= max_len):
            continue
        text_phrase = " ".join(phrase)
        score = sum(word_score[w] for w in phrase)
        if text_phrase not in scored or score > scored[text_phrase]:
            scored[text_phrase] = score

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)

    # Drop phrases that are substrings of a higher-ranked phrase with the same head.
    kept: list[tuple[str, float]] = []
    for phrase, score in ranked:
        if any(phrase in other and phrase != other for other, _ in kept):
            continue
        kept.append((phrase, score))
        if len(kept) >= top_k:
            break
    return kept
