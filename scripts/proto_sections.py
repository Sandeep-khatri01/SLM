"""Prototype: stance keywords from Wikipedia pro/con sections.

For each article, sections whose headings look like support/criticism are
collected. A global log-odds lexicon is learned from those sections across all
topics, then used to score candidate words in each article.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter

API = "https://en.wikipedia.org/w/api.php"
UA = "keyword-stance/0.1 (course project)"

PRO_HINTS = ["benefit", "advantage", "support", "arguments for", "proponent",
             "positive", "opportunit", "strength", "upside", "case for"]
CON_HINTS = ["critic", "concern", "controvers", "opposition", "arguments against",
             "drawback", "disadvantage", "risk", "challenge", "downside", "case against",
             "reception", "backlash"]

STOP = set("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves may might must shall upon within without
many much often however therefore thus since although though among across per via etc
also first second third one two three new used use using including included include
""".split())

WORD = re.compile(r"[a-z][a-z'\-]{2,}")
HEAD = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$")


def fetch_wikitext(page: str) -> str:
    q = urllib.parse.urlencode({"action": "parse", "page": page, "prop": "wikitext",
                                "format": "json", "redirects": 1})
    req = urllib.request.Request(API + "?" + q, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["parse"]["wikitext"]["*"]


def strip_markup(text: str) -> str:
    text = re.sub(r"<ref[^>]*/>", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("'''", "").replace("''", "")
    return text


def split_sections(wikitext: str) -> list[tuple[int, str, str]]:
    sections: list[tuple[int, str, str]] = []
    heading, level, body = "", 2, []
    for line in wikitext.splitlines():
        m = HEAD.match(line)
        if m:
            sections.append((level, heading, "\n".join(body)))
            level, heading, body = len(m.group(1)), m.group(2).strip(), []
        else:
            body.append(line)
    sections.append((level, heading, "\n".join(body)))
    return sections


def classify(heading: str) -> str:
    h = heading.lower()
    if any(x in h for x in PRO_HINTS):
        return "pro"
    if any(x in h for x in CON_HINTS):
        return "con"
    return "neutral"


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def main() -> None:
    pages = ["Nuclear power", "Universal basic income", "Regulation of artificial intelligence",
             "Aadhaar", "Kids Online Safety Act", "Remote work"]

    pro_counts: Counter[str] = Counter()
    con_counts: Counter[str] = Counter()
    docs = {}
    for page in pages:
        wt = fetch_wikitext(page)
        secs = split_sections(wt)
        pro_text, con_text = [], []
        found = []
        for level, heading, body in secs:
            label = classify(heading)
            if label == "neutral":
                continue
            clean = strip_markup(body)
            found.append((label, heading, len(clean)))
            (pro_text if label == "pro" else con_text).append(clean)
        pro_counts.update(words(" ".join(pro_text)))
        con_counts.update(words(" ".join(con_text)))
        docs[page] = {"pro": " ".join(pro_text), "con": " ".join(con_text), "sections": found}
        print(f"{page}: {len(found)} stance sections -> " +
              ", ".join(f"{lab}:{head}" for lab, head, _ in found[:6]))

    vocab = set(pro_counts) | set(con_counts)
    n_pro, n_con = sum(pro_counts.values()), sum(con_counts.values())
    alpha = 0.1
    lexicon = {}
    for w in vocab:
        p = (pro_counts[w] + alpha) / (n_pro + alpha * len(vocab))
        c = (con_counts[w] + alpha) / (n_con + alpha * len(vocab))
        lexicon[w] = math.log(p / c)

    for page, doc in docs.items():
        full = strip_markup(doc["pro"] + " " + doc["con"])
        freq = Counter(words(full))
        scored = [(freq[w] * lexicon.get(w, 0.0), w) for w in freq if freq[w] >= 2]
        scored.sort(reverse=True)
        pro = [w for s, w in scored if s > 0][:10]
        con = [w for s, w in sorted(scored) if s < 0][:10]
        print(f"\n== {page}")
        print("  PRO:", ", ".join(pro))
        print("  CON:", ", ".join(con))


if __name__ == "__main__":
    main()
