"""Lightweight web search with bounded retries and classified failures.

Uses DDGS (DuckDuckGo) - no API key, no cost. Transient DNS/timeout/429 errors
are retried with exponential backoff + jitter; everything else fails fast with
an actionable message the agent can show to the model and the user.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List

from ddgs import DDGS

MAX_RESULTS = 8


class SearchError(RuntimeError):
    """Search failed in a way the caller can explain to the model/user."""

    def __init__(self, message: str, reason: str = "error", hint: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        self.hint = hint

    def payload(self) -> Dict[str, str]:
        return {"error": str(self), "reason": self.reason, "hint": self.hint}


def _classify(exc: BaseException) -> str:
    text = repr(exc).lower()
    if any(
        marker in text
        for marker in (
            "name resolution",
            "nameresolution",
            "getaddrinfo",
            "temporary failure in name resolution",
            "nodename nor servname",
        )
    ):
        return "dns"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "403" in text or "blocked" in text or "captcha" in text:
        return "blocked"
    return "error"


_HINTS = {
    "dns": "Check your network/VPN, then retry with /retry or ask again.",
    "rate_limit": "Wait a few seconds, then retry; or switch search provider.",
    "timeout": "The search backend was slow. Retry with a narrower query.",
    "blocked": "The backend blocked the request. Try a different query or provider.",
    "empty": "Try a broader query or a different search backend.",
    "error": "Retry, or answer without web search and cite what you know.",
}


def web_search(query: str, max_results: int = 5, timeout: float = 8.0, attempts: int = 2) -> List[Dict[str, str]]:
    query = (query or "").strip()
    if not query:
        raise SearchError("Empty search query.", "empty", _HINTS["empty"])
    max_results = max(1, min(int(max_results or 5), MAX_RESULTS))

    last: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            rows = DDGS(timeout=timeout).text(query, max_results=max_results, backend="auto")
            results: List[Dict[str, str]] = []
            for row in rows or []:
                title = (row.get("title") or "").strip()
                href = (row.get("href") or row.get("url") or "").strip()
                body = (row.get("body") or row.get("snippet") or "").strip()
                if not href and not title:
                    continue
                results.append({"title": title, "href": href, "body": body})
            if not results:
                raise SearchError(
                    f"No results for {query!r}.",
                    "empty",
                    _HINTS["empty"],
                )
            return results[:max_results]
        except SearchError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            reason = _classify(exc)
            if reason in ("dns", "timeout", "rate_limit") and attempt < attempts - 1:
                delay = min(4.0, 0.6 * (2**attempt)) * (0.75 + random.random() * 0.5)
                time.sleep(delay)
                continue
            raise SearchError(
                f"Search failed ({reason}): {exc}",
                reason,
                _HINTS.get(reason, _HINTS["error"]),
            ) from exc

    raise SearchError(f"Search failed: {last}", _classify(last or Exception()), _HINTS["error"])


def format_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return ""
    lines = [
        "Web search results (use these for up-to-date facts; cite sources by URL):",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title','')}")
        lines.append(f"    URL: {r.get('href','')}")
        if r.get("body"):
            lines.append(f"    Snippet: {r['body']}")
    lines.append(
        "\nWhen you use these results, include a short 'Sources:' section with the URLs you relied on."
    )
    return "\n".join(lines)
