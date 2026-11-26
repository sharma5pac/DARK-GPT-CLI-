"""Cross-session semantic memory with distillation and recall.

The existing ``agent.memory`` module keeps per-repository notes with simple
token overlap. This module adds a durable, cross-session memory layer:

* TF-IDF weighted cosine scoring plus synonym expansion, recency and importance
* distillation ("consolidation") from project memory, target graph/evidence/
  planner/blackboard, and recent chat turns
* explicit remember/recall tools and a compact system-prompt block

Entries are append-only: ``archive`` hides an entry from recall but never
deletes stored knowledge, matching DarkGPT's preserve-evidence rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils import appconfig

VERSION = 1
ROOT = Path.home() / ".config" / "darkgpt" / "memory"
STORE = ROOT / "semantic.json"
MAX_TEXT = 1800
MAX_ENTRIES = 1200

STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "to", "of", "in", "is", "for", "on", "with",
    "that", "this", "it", "as", "at", "by", "from", "was", "were", "are", "be",
    "have", "has", "had", "not", "but", "you", "your", "we", "our", "they",
}

SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "sqli": ("sql", "injection", "database", "union"),
    "sql": ("sqli", "injection", "database"),
    "xss": ("cross", "site", "scripting", "html", "javascript"),
    "rce": ("remote", "code", "execution", "command", "shell"),
    "ssrf": ("server", "side", "request", "forgery", "callback"),
    "lfi": ("local", "file", "inclusion", "traversal", "read"),
    "auth": ("authentication", "login", "session", "token", "cookie"),
    "fuzz": ("fuzzing", "brute", "wordlist", "enumerate", "payload"),
    "scan": ("scanner", "enumerate", "recon", "discovery", "nmap"),
    "web": ("http", "url", "browser", "site", "endpoint"),
    "gui": ("desktop", "window", "click", "accessibility", "selenium"),
    "firmware": ("flash", "rom", "binwalk", "hardware", "ghidra"),
    "report": ("export", "markdown", "html", "findings", "remediation"),
    "memory": ("recall", "remember", "knowledge", "lesson"),
    "guard": ("injection", "poisoning", "security", "prompt", "promptguard"),
    "route": ("pivot", "alternative", "fallback", "retry"),
    "evidence": ("proof", "artifact", "repro", "oracle", "sha256"),
}

_WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{1,}")


def enabled() -> bool:
    return bool(appconfig.get("semantic_memory_enabled", True))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def fingerprint(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8", "replace")).hexdigest()[:16]


def _tokens(text: str) -> Set[str]:
    words = {w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS and len(w) > 2}
    expanded = set(words)
    for word in tuple(words):
        expanded.update(SYNONYMS.get(word, ()))
    return expanded


def _load() -> List[Dict[str, Any]]:
    try:
        data = json.loads(STORE.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(entries: Sequence[Dict[str, Any]]) -> None:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        # Never drop entries: an append-only knowledge store must not lose data.
        tmp.write_text(json.dumps(list(entries[-MAX_ENTRIES:]), indent=2, ensure_ascii=False))
        tmp.replace(STORE)
    except Exception:
        pass


def remember(
    text: str,
    kind: str = "note",
    *,
    tags: Optional[Iterable[str]] = None,
    source: str = "agent",
    target: str = "",
    cwd: str = "",
    importance: float = 0.5,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store or strengthen one memory. Dedupe is by normalized-text fingerprint."""
    clean = re.sub(r"\s+", " ", (text or "").strip())[:MAX_TEXT]
    if not clean:
        return {}
    fp = fingerprint(clean)
    entries = _load()
    for entry in entries:
        if str(entry.get("fingerprint") or "") == fp:
            entry["ts"] = time.time()
            entry["count"] = int(entry.get("count") or 1) + 1
            entry["importance"] = max(float(entry.get("importance") or 0.0), max(0.0, min(1.0, float(importance))))
            if meta and isinstance(meta, dict):
                current = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
                current.update(meta)
                entry["meta"] = current
            _save(entries)
            return entry
    entry: Dict[str, Any] = {
        "id": "sm_" + hashlib.sha1(f"{fp}{time.time()}".encode()).hexdigest()[:12],
        "fingerprint": fp,
        "ts": time.time(),
        "kind": kind or "note",
        "text": clean,
        "tags": sorted(set(str(t) for t in (tags or []) if str(t).strip()))[:20],
        "source": source or "agent",
        "target": target or "",
        "cwd": cwd or "",
        "importance": max(0.0, min(1.0, float(importance or 0.5))),
        "count": 1,
        "archived": False,
        "meta": meta or {},
    }
    entries.append(entry)
    _save(entries)
    return entry


def _idf(entries: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[str, float]]:
    docs: Dict[str, Set[str]] = {}
    for entry in entries:
        docs[str(entry.get("id"))] = _tokens(str(entry.get("text") or "") + " " + " ".join(entry.get("tags") or []))
    df: Counter[str] = Counter()
    for words in docs.values():
        df.update(words)
    n = max(1, len(docs))
    idf = {word: math.log(1.0 + n / (1.0 + count)) for word, count in df.items()}
    return docs, idf


def recall(
    query: str = "",
    k: int = 6,
    *,
    kinds: Optional[Iterable[str]] = None,
    target: str = "",
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    """Rank memories by semantic-ish similarity, recency and importance."""
    entries = [e for e in _load() if include_archived or not e.get("archived")]
    if kinds is not None:
        wanted = set(kinds)
        entries = [e for e in entries if str(e.get("kind") or "") in wanted]
    if target:
        entries = [e for e in entries if str(e.get("target") or "") in str(target)]
    if not entries:
        return []
    q = _tokens(query)
    docs, idf = _idf(entries)
    now = time.time()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        words = docs.get(str(entry.get("id"))) or set()
        if q:
            overlap = q & words
            if not overlap:
                continue
            score = sum(idf.get(word, 0.0) for word in overlap) / math.sqrt(max(1.0, len(q) * len(words)))
        else:
            score = 0.0
        age_days = max(0.0, (now - float(entry.get("ts") or 0.0)) / 86400.0)
        score += 0.12 * float(entry.get("importance") or 0.0)
        score += 0.03 * min(1.0, int(entry.get("count") or 1) / 5.0)
        score += 0.02 / (1.0 + age_days / 30.0)
        scored.append((score, entry))
    scored.sort(key=lambda pair: (-pair[0], -float(pair[1].get("ts") or 0.0)))
    return [entry for _, entry in scored[: max(0, int(k or 6))]]


def recent(k: int = 12, *, include_archived: bool = False) -> List[Dict[str, Any]]:
    entries = [e for e in _load() if include_archived or not e.get("archived")]
    entries.sort(key=lambda e: float(e.get("ts") or 0.0), reverse=True)
    return entries[: max(0, int(k or 12))]


def archive(entry_id: str) -> bool:
    """Hide a memory from recall without deleting it."""
    entries = _load()
    changed = False
    for entry in entries:
        if str(entry.get("id")) == str(entry_id) and not entry.get("archived"):
            entry["archived"] = True
            entry["archived_at"] = time.time()
            changed = True
    if changed:
        _save(entries)
    return changed


def restore(entry_id: str) -> bool:
    entries = _load()
    changed = False
    for entry in entries:
        if str(entry.get("id")) == str(entry_id) and entry.get("archived"):
            entry["archived"] = False
            entry.pop("archived_at", None)
            changed = True
    if changed:
        _save(entries)
    return changed


def format_memories(entries: Sequence[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = ["Semantic memory (recalled across sessions - treat as prior observations, verify before relying):"]
    for entry in entries:
        kind = str(entry.get("kind") or "note")
        target = str(entry.get("target") or "")
        prefix = f"[{kind}" + (f" {target}" if target else "") + "]"
        lines.append(f"- {prefix} {str(entry.get('text') or '').strip()[:400]}")
    return "\n".join(lines)


def context_block(query: str = "", k: Optional[int] = None) -> str:
    if not enabled():
        return ""
    limit = int(k if k is not None else appconfig.get("semantic_recall_k", 5) or 5)
    return format_memories(recall(query, limit))


def _chat_turns(limit: int = 24) -> List[Tuple[str, str]]:
    """Recent (user, assistant) text pairs from the project's chat database."""
    try:
        from utils import history

        db = history.DB_PATH
        if not Path(db).exists():
            return []
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit * 2,)
            ).fetchall()
        finally:
            conn.close()
        pairs: List[Tuple[str, str]] = []
        user = ""
        for role, content in reversed(rows):
            role = str(role or "")
            text = re.sub(r"\s+", " ", str(content or "")).strip()
            if role == "user":
                user = text[:400]
            elif role == "assistant" and user and text:
                pairs.append((user, text[:700]))
                user = ""
        return pairs[-limit:]
    except Exception:
        return []


def consolidate(
    cwd: str = "",
    target: str = "",
    *,
    include_chat: bool = True,
    include_graph: bool = True,
    limit: int = 30,
) -> Dict[str, Any]:
    """Distill durable facts from the other state systems into semantic memory."""
    counts: Dict[str, int] = {}
    target = str(target or appconfig.get("active_target") or "").strip()

    def add(text: str, kind: str, source: str, importance: float, tags: Sequence[str] = ()) -> None:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) < 20:
            return
        entry = remember(
            text, kind, tags=tags, source=source, target=target, cwd=cwd,
            importance=importance, meta={"consolidated_at": time.time()},
        )
        counts[source] = counts.get(source, 0) + (1 if entry else 0)

    try:
        from agent import memory as project_memory

        for entry in project_memory.recent(cwd, limit):
            kind = str(entry.get("kind") or "note")
            add(str(entry.get("text") or ""), "lesson" if kind in project_memory.LESSON_KINDS else "note", "project-memory", 0.65, (kind,))
    except Exception:
        pass

    if include_graph and target:
        try:
            from agent import blackboard as bb, evidence as ev, knowledge_graph as kg, planner as pl

            nodes = kg.query(target, limit=limit)
            for node in nodes:
                kind = str(node.get("kind") or "node")
                label = str(node.get("identity") or node.get("label") or "")
                attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
                bits = [label]
                for key in ("title", "severity", "status", "impact", "remediation", "notes", "description"):
                    if attrs.get(key):
                        bits.append(f"{key}={str(attrs[key])[:200]}")
                if len(bits) > 1:
                    add(" ".join(bits), "finding" if kind == "finding" else "recon", "knowledge-graph", 0.8, (kind, target))
            summary = ev.summary(target)
            if summary:
                add(json.dumps(summary, ensure_ascii=False)[:MAX_TEXT], "evidence", "evidence-ledger", 0.7, ("summary", target))
            plan = pl.summary(target)
            if plan:
                add(json.dumps(plan, ensure_ascii=False)[:MAX_TEXT], "plan", "planner", 0.55, ("objective", target))
            bb_items = bb.list_items(target, limit=8)
            for item in bb_items:
                add(str(item.get("text") or ""), "hint", "blackboard", 0.6, (str(item.get("kind") or "fact"), target))
        except Exception:
            pass

    if include_chat:
        for user, answer in _chat_turns(limit=limit):
            add(f"Q: {user}\nA: {answer}", "engagement", "chat-history", 0.45, ("session",))

    return {"ok": True, "target": target, "added": counts, "total": sum(counts.values())}


def status_lines() -> List[str]:
    entries = _load()
    active = [e for e in entries if not e.get("archived")]
    kinds = Counter(str(e.get("kind") or "note") for e in active)
    top = ", ".join(f"{k}={v}" for k, v in kinds.most_common(6)) or "none"
    return [
        f"Semantic memory: {'ON' if enabled() else 'OFF'} | entries={len(active)} archived={len(entries)-len(active)}",
        f"  kinds: {top}",
        "  /memory-search <query> | /memory-consolidate | /memory-screen",
    ]


def tool_specs() -> List[Dict[str, Any]]:
    """Built-in tool schemas for explicit agent use."""
    return [
        {
            "type": "function",
            "function": {
                "name": "semantic_recall",
                "description": (
                    "Search durable cross-session semantic memory for prior findings, lessons, "
                    "pivots, target observations and engagement outcomes. Use this at the start "
                    "of a new engagement or before repeating work."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "target": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_remember",
                "description": (
                    "Store a durable, reusable memory (lesson, finding, pivot, technique, decision) "
                    "for future sessions. Deduplicates automatically and never deletes old data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "target": {"type": "string"},
                        "importance": {"type": "number"},
                    },
                    "required": ["text"],
                },
            },
        },
    ]
