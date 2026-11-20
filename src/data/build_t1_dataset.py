"""Build large-scale stance keyword dataset (T1 pipeline scaffolding).

Harvests debate arguments from Wikipedia controversy/benefits sections and
 debate resources, extracting stance keywords via RAKE (no neural distillation).

Produces structured JSONL ready for BPE tokenization and T1 scaling:
    {"topic": str, "stance": "positive"|"negative", "keywords": list[str]}

Usage:
    python -m src.data.build_t1_dataset --pages "Nuclear power" "Electric cars" --out data/t1/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from src.data.keywords import extract_keywords

API = "https://en.wikipedia.org/w/api.php"
UA = "keyword-stance/0.2 (open-domain research project)"

PRO_HINTS = [
    "benefit", "advantage", "support", "arguments for", "proponent",
    "positive", "opportunity", "strength", "upside", "case for", "rationale"
]
CON_HINTS = [
    "critic", "concern", "controvers", "opposition", "arguments against",
    "drawback", "disadvantage", "risk", "challenge", "downside", "case against",
    "reception", "backlash", "harms", "danger"
]


def strip_markup(text: str) -> str:
    text = re.sub(r"<ref[^>]*/>", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text.replace("'''", "").replace("''", "")


def split_sections(wikitext: str) -> list[tuple[str, str]]:
    """Return list of (heading, body_text)."""
    head_pattern = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$")
    sections = []
    heading, body = "Introduction", []
    for line in wikitext.splitlines():
        m = head_pattern.match(line)
        if m:
            sections.append((heading, "\n".join(body)))
            heading, body = m.group(2).strip(), []
        else:
            body.append(line)
    sections.append((heading, "\n".join(body)))
    return sections


def fetch_wikitext(page: str) -> str:
    q = urllib.parse.urlencode({
        "action": "parse",
        "page": page,
        "prop": "wikitext",
        "format": "json",
        "redirects": 1,
    })
    req = urllib.request.Request(API + "?" + q, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
        if "error" in data:
            return ""
        return data.get("parse", {}).get("wikitext", {}).get("*", "")


def harvest_topic(topic_title: str) -> list[dict]:
    wt = fetch_wikitext(topic_title)
    if not wt:
        return []

    sections = split_sections(wt)
    pro_texts: list[str] = []
    con_texts: list[str] = []

    for heading, body in sections:
        h_lower = heading.lower()
        clean_body = strip_markup(body)
        if any(h in h_lower for h in PRO_HINTS):
            pro_texts.append(clean_body)
        elif any(h in h_lower for h in CON_HINTS):
            con_texts.append(clean_body)

    records: list[dict] = []
    if pro_texts:
        pro_combined = "\n".join(pro_texts)
        pro_kws = [kw for kw, _ in extract_keywords(pro_combined, top_k=10, max_len=2)]
        if pro_kws:
            records.append({
                "topic": topic_title,
                "stance": "positive",
                "keywords": pro_kws,
            })

    if con_texts:
        con_combined = "\n".join(con_texts)
        con_kws = [kw for kw, _ in extract_keywords(con_combined, top_k=10, max_len=2)]
        if con_kws:
            records.append({
                "topic": topic_title,
                "stance": "negative",
                "keywords": con_kws,
            })

    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pages", nargs="+", required=True, help="Wikipedia page titles")
    p.add_argument("--out", required=True, help="Output JSONL path")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for page in args.pages:
            print(f"Harvesting: {page}...")
            records = harvest_topic(page)
            for rec in records:
                f.write(json.dumps(rec) + "\n")
                total_records += 1
            time.sleep(0.5)

    print(f"✓ Done. Harvested {total_records} records to {out_path}")


if __name__ == "__main__":
    main()
