"""Fetch plain-text Wikipedia extracts and links (stdlib only, rate-limit aware)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
UA = "keyword-stance/0.1 (course project; contact: local)"


def _api(params: dict) -> dict:
    """GET the API with retries on 429/5xx and Retry-After support."""
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2**attempt, 30)
            time.sleep(wait)
    raise RuntimeError(f"Wikipedia API failed after retries: {last_error}")


def fetch_extract(title: str, max_chars: int = 60_000) -> tuple[str, str]:
    """Return (resolved_title, plain_text). Empty text if the page is missing."""
    data = _api(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
    )
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if "missing" in page:
        return title, ""
    return page.get("title", title), page.get("extract", "")[:max_chars]


def fetch_links(title: str, max_links: int = 500) -> tuple[str, list[str]]:
    """Return (resolved_title, linked article titles) for the main namespace."""
    data = _api(
        {
            "action": "query",
            "prop": "links",
            "plnamespace": 0,
            "pllimit": "max",
            "redirects": 1,
            "titles": title,
        }
    )
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if "missing" in page:
        return title, []
    links = [link["title"] for link in page.get("links", []) if link.get("ns") == 0]
    return page.get("title", title), links[:max_links]
