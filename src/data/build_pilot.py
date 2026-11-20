"""Pilot: topic -> Wikipedia text -> candidate keywords.

Run from repository root:

    python -m src.data.build_pilot --limit 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.keywords import extract_keywords
from src.data.wiki import fetch_extract, fetch_links

GENERIC = {
    "isbn", "doi", "pmid", "pmc", "arxiv", "jstor", "wayback machine", "bibcode",
    "ssrn", "oclc", "issn", "hdl", "s2cid", "official website", "worldcat",
}


def clean_links(links: list[str], topic_title: str, limit: int = 25) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for link in links:
        key = link.lower()
        if key in seen or key in GENERIC or key == topic_title.lower():
            continue
        if len(key) < 3 or key.isdigit():
            continue
        seen.add(key)
        out.append(link)
        if len(out) >= limit:
            break
    return out


def rank_links(text: str, links: list[str], limit: int = 20) -> list[str]:
    """Rank linked titles by how often they appear in the article text.

    The links API returns titles alphabetically, so frequency + first position is
    the cheapest useful relevance signal.
    """
    lower = text.lower()
    scored = []
    for link in links:
        title = link.lower()
        count = lower.count(title)
        if count == 0:
            continue
        scored.append((count, lower.find(title), link))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [link for _, _, link in scored[:limit]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", default="data/topics.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", default="data/pilot/keywords.jsonl")
    args = p.parse_args()

    topics = [json.loads(line) for line in Path(args.topics).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        topics = topics[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in topics:
            title, text = fetch_extract(item["wiki"])
            if not text:
                print(f"!! no extract for {item['topic']} ({item['wiki']})")
                continue
            _, links = fetch_links(item["wiki"])
            link_keywords = rank_links(text, clean_links(links, title, limit=500))
            keywords = extract_keywords(text, top_k=args.top_k)
            record = {
                "topic": item["topic"],
                "domain": item.get("domain"),
                "wiki_title": title,
                "chars": len(text),
                "link_keywords": link_keywords,
                "rake_keywords": [{"phrase": k, "score": round(s, 3)} for k, s in keywords],
            }
            f.write(json.dumps(record) + "\n")
            print(f"\n== {item['topic']}  ({len(text):,} chars, {len(links)} links)")
            print("   links: " + ", ".join(link_keywords[:15]))
            print("   rake : " + ", ".join(k for k, _ in keywords[:10]))


if __name__ == "__main__":
    main()
