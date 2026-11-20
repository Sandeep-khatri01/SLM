"""Generate a tiny synthetic corpus in the training format.

This exists only to smoke-test the training pipeline end to end before the real
keyword corpus exists. The associations it teaches are meaningless.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TOPICS = [
    "ai regulation", "nuclear energy", "universal basic income", "remote work",
    "gene editing", "carbon tax", "four day workweek", "autonomous vehicles",
    "school uniforms", "standardized testing", "social media age limits",
    "lab grown meat", "open source ai", "space exploration funding",
    "electric vehicle subsidies", "rent control", "free college tuition",
    "minimum wage increase", "digital privacy laws", "nuclear waste storage",
    "animal testing", "smart city surveillance", "cryptocurrency regulation",
    "climate geoengineering",
]

POSITIVE = [
    "innovation", "safety", "transparency", "growth", "opportunity", "trust",
    "efficiency", "resilience", "fairness", "progress", "accountability",
    "access", "reliability", "benefit", "protection", "investment", "equity",
    "clarity", "stability", "choice",
]

NEGATIVE = [
    "bureaucracy", "cost", "risk", "stagnation", "burden", "inequality",
    "surveillance", "delay", "complexity", "harm", "uncertainty", "compliance",
    "lobbying", "shortage", "dependency", "erosion", "backlash", "waste",
    "exclusion", "instability",
]


def main() -> None:
    rng = random.Random(7)
    out = Path("data/dummy/train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for topic in TOPICS:
            topic_words = topic.split()
            for stance, pool in (("positive", POSITIVE), ("negative", NEGATIVE)):
                for _ in range(20):
                    k = rng.randint(5, 8)
                    words = rng.sample(pool, k - len(topic_words))
                    keywords = topic_words + words
                    rng.shuffle(keywords)
                    record = {"topic": topic, "stance": stance, "keywords": keywords, "split": "train"}
                    f.write(json.dumps(record) + "\n")
                    count += 1
    print(f"wrote {count} records to {out}")


if __name__ == "__main__":
    main()
