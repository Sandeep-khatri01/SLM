"""T0 retrieval fallback: map any user input to nearest known topics.

Uses TF-IDF with character n-grams (2–4) over topic names and their keyword
sets.  When the user types a topic not in the closed set, the index finds the
closest matches and returns their keywords merged, with a similarity score
so the UI can warn when confidence is low.

This module has no model dependency — it runs on CPU in <1 ms per query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalHit:
    topic: str
    domain: str
    similarity: float
    pro: list[str]
    con: list[str]


# Common paraphrases / aliases so the char n-gram vectoriser can match them.
_ALIASES: dict[str, list[str]] = {
    "autonomous vehicles": ["driverless cars", "self driving", "self-driving cars", "robo-taxi"],
    "AI regulation": ["artificial intelligence regulation", "AI rules", "AI governance", "AI law"],
    "AI in healthcare": ["medical AI", "healthcare AI", "clinical AI"],
    "AI in education": ["educational AI", "edtech AI", "AI tutoring"],
    "AI art copyright": ["AI generated art", "generative art copyright", "AI image rights"],
    "AI development pause": ["AI moratorium", "pause AI", "halt AI training"],
    "AI weapons": ["lethal autonomous weapons", "killer robots", "military AI"],
    "AI job displacement": ["AI unemployment", "AI automation jobs", "robots taking jobs"],
    "AI hiring tools": ["AI recruitment", "algorithmic hiring"],
    "AI companions": ["AI chatbot companions", "AI friends", "virtual companions"],
    "facial recognition": ["face recognition", "face scanning", "facial surveillance"],
    "universal basic income": ["UBI", "basic income guarantee"],
    "four-day workweek": ["four day work week", "4 day workweek", "shorter workweek"],
    "encryption backdoors": ["encryption back doors", "government encryption access"],
    "phone bans in schools": ["school phone ban", "mobile phones schools", "school phones"],
    "social media age limits": ["age verification social media", "children social media"],
    "demonetisation": ["demonetization", "note ban"],
    "EV subsidies": ["electric vehicle subsidies", "EV incentives"],
    "content moderation": ["platform moderation", "online content rules"],
    "loot boxes": ["loot box regulation", "gaming microtransactions"],
    "dark patterns": ["deceptive design", "manipulative UX"],
    "deepfake rules": ["deepfake regulation", "deepfake law", "synthetic media rules"],
    "data protection act": ["data privacy law", "DPDP", "privacy regulation"],
    "gig worker classification": ["gig economy workers", "freelancer classification"],
    "remote work": ["work from home", "WFH", "telecommuting"],
    "minimum wage increase": ["higher minimum wage", "wage hike", "raise minimum wage"],
    "resume screening AI": ["AI resume screening", "automated resume review"],
    "open-source AI": ["open source artificial intelligence", "OSS AI"],
    "ransomware payment ban": ["ransomware ban", "ban ransomware payments"],
    "bug bounty mandates": ["vulnerability disclosure", "bug bounty program"],
    "cyber insurance mandates": ["cybersecurity insurance"],
    "biometric authentication": ["biometric login", "fingerprint authentication"],
    "data breach disclosure": ["breach notification", "data leak disclosure"],
    "government hacking powers": ["state hacking", "government surveillance hacking"],
    "IoT security regulation": ["internet of things security", "smart device regulation"],
    "gambling ads": ["gambling advertising", "betting ads"],
    "influencer marketing": ["influencer advertising", "sponsored content"],
    "short-form video bans": ["TikTok ban", "short video ban", "reels ban"],
    "digital detox": ["screen time reduction", "technology detox"],
    "automation tax": ["robot tax", "automation levy"],
    "farm laws": ["agriculture reform", "farm bills", "APMC reform"],
    "semiconductor mission": ["chip manufacturing", "semiconductor fab"],
    "Article 370": ["Kashmir autonomy", "J&K special status"],
    "CAA": ["citizenship amendment", "citizenship act"],
    "MSP": ["minimum support price", "crop price guarantee"],
    "GST": ["goods and services tax"],
    "Aadhaar": ["aadhaar card", "UID", "unique identification"],
    "UPI": ["unified payments interface", "digital payments india"],
}


def _build_document(entry: dict) -> str:
    """Build a rich text document from a topic entry for TF-IDF indexing.

    Combines the topic name (repeated for weight), domain, aliases, and all
    keywords so that both the topic string and its semantic keyword
    neighbourhood are searchable.
    """
    parts = [
        entry["topic"],
        entry["topic"],       # repeat for extra weight
        entry["topic"],       # triple weight on the topic name
        entry.get("domain", ""),
    ]
    # Inject aliases so paraphrases match via n-grams
    for alias in _ALIASES.get(entry["topic"], []):
        parts.append(alias)
    for kw in entry.get("pro", []):
        parts.append(kw)
    for kw in entry.get("con", []):
        parts.append(kw)
    return " ".join(parts).lower()


class TopicIndex:
    """TF-IDF index over the 48 known topics."""

    def __init__(self, topics_path: str | Path) -> None:
        self.entries: list[dict] = []
        path = Path(topics_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                self.entries.append(json.loads(line))

        self.topic_names_lower = [e["topic"].lower() for e in self.entries]

        # Build TF-IDF matrix using character n-grams (robust to typos)
        # plus word-level unigrams/bigrams
        documents = [_build_document(e) for e in self.entries]

        # Character-level n-gram vectoriser (handles typos, partial matches)
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=8000,
        )
        self._char_matrix = self._char_vec.fit_transform(documents)

        # Word-level vectoriser (handles semantic overlap via shared keywords)
        self._word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=4000,
        )
        self._word_matrix = self._word_vec.fit_transform(documents)

    def is_exact(self, query: str) -> bool:
        return query.strip().lower() in self.topic_names_lower

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        """Find the top-k most similar topics to the query string."""
        q = query.strip().lower()

        # Exact match — return it alone with score 1.0
        if q in self.topic_names_lower:
            idx = self.topic_names_lower.index(q)
            e = self.entries[idx]
            return [RetrievalHit(
                topic=e["topic"],
                domain=e["domain"],
                similarity=1.0,
                pro=e.get("pro", []),
                con=e.get("con", []),
            )]

        # Compute similarities from both vectorisers, blend 40% char + 60% word
        char_q = self._char_vec.transform([q])
        word_q = self._word_vec.transform([q])

        char_sim = cosine_similarity(char_q, self._char_matrix).flatten()
        word_sim = cosine_similarity(word_q, self._word_matrix).flatten()

        blended = 0.4 * char_sim + 0.6 * word_sim

        top_indices = blended.argsort()[::-1][:top_k]
        results: list[RetrievalHit] = []
        for idx in top_indices:
            score = float(blended[idx])
            if score < 0.01:
                continue
            e = self.entries[idx]
            results.append(RetrievalHit(
                topic=e["topic"],
                domain=e["domain"],
                similarity=round(score, 4),
                pro=e.get("pro", []),
                con=e.get("con", []),
            ))
        return results

    def merge_results(self, hits: list[RetrievalHit]) -> dict:
        """Merge keyword sets from multiple hits, weighted by similarity."""
        if not hits:
            return {"pro": [], "con": [], "sources": [], "confidence": 0.0}

        # Deduplicate keywords while preserving order by first appearance
        seen_pro: set[str] = set()
        seen_con: set[str] = set()
        pro: list[str] = []
        con: list[str] = []
        sources: list[dict] = []

        for hit in hits:
            sources.append({
                "topic": hit.topic,
                "domain": hit.domain,
                "similarity": hit.similarity,
            })
            for kw in hit.pro:
                if kw.lower() not in seen_pro:
                    seen_pro.add(kw.lower())
                    pro.append(kw)
            for kw in hit.con:
                if kw.lower() not in seen_con:
                    seen_con.add(kw.lower())
                    con.append(kw)

        # Cap at 10 keywords per side
        pro = pro[:10]
        con = con[:10]

        return {
            "pro": pro,
            "con": con,
            "sources": sources,
            "confidence": hits[0].similarity if hits else 0.0,
        }
