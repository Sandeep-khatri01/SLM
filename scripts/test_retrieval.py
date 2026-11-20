"""Quick smoke test for the T0 retrieval fallback."""
import sys
sys.path.insert(0, ".")

from src.retrieval import TopicIndex

index = TopicIndex("data/curated/topics.jsonl")

tests = [
    "AI regulation",          # exact match → score 1.0
    "artificial intelligence", # paraphrase
    "quantum computing",       # out-of-domain
    "lab-grown meat",          # out-of-domain
    "crypto regulation",       # related to regulation topics
    "school phones",           # paraphrase of phone bans
    "driverless cars",         # paraphrase of autonomous vehicles
    "online gambling",         # related to loot boxes / gambling ads
    "climate change tax",      # related to EV subsidies, carbon topics
]

for query in tests:
    hits = index.search(query, top_k=3)
    print(f"\n{'='*60}")
    print(f"Query: {query!r}")
    if hits and hits[0].similarity == 1.0:
        print(f"  → EXACT MATCH: {hits[0].topic}")
    else:
        for h in hits:
            print(f"  → {h.topic} ({h.domain}) — {h.similarity:.4f}")
        merged = index.merge_results(hits)
        print(f"  PRO: {merged['pro'][:5]}")
        print(f"  CON: {merged['con'][:5]}")
        print(f"  Confidence: {merged['confidence']:.4f}")
