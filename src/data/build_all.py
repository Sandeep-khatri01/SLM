"""Build train files that cover all curated topics (for the final demo models)."""

from __future__ import annotations

import json
import random
from pathlib import Path

VARIANTS = 30


def main() -> None:
    rng = random.Random(7)
    topics = [
        json.loads(line)
        for line in Path("data/curated/topics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for stance, key, label in (("pro", "pro", "positive"), ("con", "con", "negative")):
        out = Path(f"data/{stance}/all.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for topic in topics:
                keywords = topic[key]
                for _ in range(VARIANTS):
                    k = rng.randint(max(4, len(keywords) - 3), len(keywords))
                    sample = rng.sample(keywords, k)
                    f.write(
                        json.dumps({"topic": topic["topic"], "stance": label, "keywords": sample})
                        + "\n"
                    )
        print(f"{out}: {len(topics) * VARIANTS} records over {len(topics)} topics")


if __name__ == "__main__":
    main()
