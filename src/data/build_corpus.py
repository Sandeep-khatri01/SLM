"""Build the pilot topic corpus and rank link-keywords with IDF.

For every topic: fetch a Wikipedia extract + its outgoing links, count how often
each linked title appears in the text, then down-weight titles that appear in
many topics (IDF). This is the candidate-keyword stage of the data pipeline.

Run from repository root:

    python -m src.data.build_corpus
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from src.data.build_pilot import clean_links
from src.data.wiki import fetch_extract, fetch_links


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    topics = [
        json.loads(line)
        for line in Path("data/topics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_dir = Path("data/pilot")
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for item in topics:
        title, text = fetch_extract(item["wiki"])
        if not text:
            print(f"!! missing: {item['wiki']}")
            continue
        time.sleep(0.5)
        _, links = fetch_links(item["wiki"])
        time.sleep(1.0)
        links = clean_links(links, title, limit=500)
        lower = text.lower()
        counts: Counter[str] = Counter()
        first: dict[str, int] = {}
        for link in links:
            count = lower.count(link.lower())
            if count:
                counts[link] = count
                first[link] = lower.find(link.lower())
        docs.append({"item": item, "title": title, "text": text, "counts": counts, "first": first})
        (docs_dir / f"{slugify(item['topic'])}.txt").write_text(text, encoding="utf-8")
        print(f"fetched {item['topic']:<28} {len(text):>7,} chars  {len(counts):>3} linked keywords")

    n = len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        for link in doc["counts"]:
            df[link] += 1

    index_path = out_dir / "corpus.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            scored = []
            for link, count in doc["counts"].items():
                idf = math.log((1 + n) / (1 + df[link])) + 1.0
                scored.append((count * idf, -doc["first"][link], link))
            scored.sort(reverse=True)
            keywords = [link for _, _, link in scored[:30]]
            record = {
                "topic": doc["item"]["topic"],
                "domain": doc["item"].get("domain"),
                "wiki_title": doc["title"],
                "slug": slugify(doc["item"]["topic"]),
                "keywords": keywords,
            }
            f.write(json.dumps(record) + "\n")
            print(f"\n== {doc['item']['topic']}")
            print("   " + ", ".join(keywords[:12]))

    print(f"\nwrote {index_path} ({n} docs)")


if __name__ == "__main__":
    main()
