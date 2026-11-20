"""Local web app for the two stance keyword models.

Run from repository root:

    python -m uvicorn app:app --port 8000

Open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data.text import decode, encode
from src.generate import load
from src.retrieval import TopicIndex

ROOT = Path(__file__).parent
WEB = ROOT / "web"

TOPICS = [
    json.loads(line)
    for line in (ROOT / "data/curated/topics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
BY_TOPIC = {t["topic"].lower(): t for t in TOPICS}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRO_MODEL, PRO_ITOS = load(str(ROOT / "outputs/pro-big/ckpt.pt"), DEVICE)
CON_MODEL, CON_ITOS = load(str(ROOT / "outputs/con-big/ckpt.pt"), DEVICE)
PRO_STOI = {w: i for i, w in enumerate(PRO_ITOS)}
CON_STOI = {w: i for i, w in enumerate(CON_ITOS)}
LOCK = threading.Lock()

# T0: retrieval index for open-domain fallback
TOPIC_INDEX = TopicIndex(ROOT / "data/curated/topics.jsonl")

app = FastAPI(title="Two Minds")
app.mount("/static", StaticFiles(directory=WEB), name="static")


class DebateRequest(BaseModel):
    topic: str


def generate_keywords(model, stoi, itos, topic: str, stance: str, n_tokens: int = 30) -> list[str]:
    prompt = f"topic: {topic.lower()} | stance: {stance} | keywords:"
    ids = encode(prompt, stoi)[:-1]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with LOCK:
        out = model.generate(x, max_new_tokens=n_tokens, temperature=0.0)
    continuation = decode(out[0, len(ids):].tolist(), itos)
    continuation = continuation.split(" topic")[0]
    keywords: list[str] = []
    for part in continuation.split(","):
        word = " ".join(part.strip(" .,:;|").split())
        if not word or word in {"and", "or", "stance", "keywords", "topic", "positive", "negative"}:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= 8:
            break
    return keywords


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


# Warm up CUDA kernels once so the first user request is not a 15-second stall.
with torch.no_grad():
    generate_keywords(PRO_MODEL, PRO_STOI, PRO_ITOS, "AI regulation", "positive", 8)
    generate_keywords(CON_MODEL, CON_STOI, CON_ITOS, "AI regulation", "negative", 8)


@app.get("/api/topics")
def topics() -> dict:
    return {"topics": [{"topic": t["topic"], "domain": t["domain"]} for t in TOPICS]}


@app.post("/api/debate")
def debate(req: DebateRequest) -> dict:
    key = req.topic.strip().lower()

    # Exact match → use the generative model
    if key in BY_TOPIC:
        item = BY_TOPIC[key]
        t0 = time.time()
        pro = generate_keywords(PRO_MODEL, PRO_STOI, PRO_ITOS, item["topic"], "positive")
        con = generate_keywords(CON_MODEL, CON_STOI, CON_ITOS, item["topic"], "negative")
        return {
            "topic": item["topic"],
            "domain": item["domain"],
            "pro": pro,
            "con": con,
            "ms": round((time.time() - t0) * 1000),
            "mode": "generative",
        }

    # T0 retrieval fallback — find nearest known topics
    t0 = time.time()
    hits = TOPIC_INDEX.search(req.topic.strip(), top_k=3)
    if not hits:
        raise HTTPException(status_code=404, detail="no similar topic found")

    merged = TOPIC_INDEX.merge_results(hits)
    return {
        "topic": req.topic.strip(),
        "domain": hits[0].domain,
        "pro": merged["pro"],
        "con": merged["con"],
        "ms": round((time.time() - t0) * 1000),
        "mode": "retrieval",
        "confidence": merged["confidence"],
        "sources": merged["sources"],
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "device": DEVICE,
        "topics": len(TOPICS),
        "params_pro": sum(p.numel() for p in PRO_MODEL.parameters()),
        "params_con": sum(p.numel() for p in CON_MODEL.parameters()),
        "retrieval_index": True,
    }
