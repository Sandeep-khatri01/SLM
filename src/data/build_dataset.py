"""Turn the curated topic sheet into train/val JSONL for both stances.

Held-out topics are never trained on; they are the only honest way to talk about
generalization in the report.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
VARIANTS = 30
HOLDOUT = 8


def main() -> None:
    rng = random.Random(SEED)
    topics = [
        json.loads(line)
        for line in Path("data/curated/topics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng.shuffle(topics)
    holdout, train = topics[:HOLDOUT], topics[HOLDOUT:]

    meta = {"train": [t["topic"] for t in train], "holdout": [t["topic"] for t in holdout]}
    Path("data/meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for stance, key, label in (("pro", "pro", "positive"), ("con", "con", "negative")):
        for split, items in (("train", train), ("val", holdout)):
            out = Path(f"data/{stance}/{split}.jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                for topic in items:
                    keywords = topic[key]
                    for _ in range(VARIANTS):
                        k = rng.randint(max(4, len(keywords) - 3), len(keywords))
                        sample = rng.sample(keywords, k)
                        f.write(
                            json.dumps(
                                {"topic": topic["topic"], "stance": label, "keywords": sample}
                            )
                            + "\n"
                        )
            print(f"{out}: {len(items) * VARIANTS} records")

    print(f"train topics={len(train)} holdout={len(holdout)}")
    print("holdout:", ", ".join(meta["holdout"]))


if __name__ == "__main__":
    main()
