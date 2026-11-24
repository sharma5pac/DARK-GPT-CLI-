"""Persistent per-repository memory.

Stores concise decisions, conventions and fixes learned during successful turns
and recalls the most relevant ones for the next prompt. Scoring is a simple
token-overlap rank - deterministic, offline and dependency-free.

Self-learning extension: lessons/patterns/failures are recorded as first-class
memory kinds with deduplication so the agent can avoid repeating mistakes and
recognise recurring patterns across sessions.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import vcs

ROOT = Path.home() / ".config" / "darkgpt" / "memory"
MAX_ENTRIES = 400
STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "for", "on", "with", "that", "this", "it", "as", "at", "by"}

LESSON_KINDS = {"lesson", "failed-approach", "pattern", "correction"}
KIND_LIMITS = {"lesson": 80, "failed-approach": 120, "pattern": 80, "correction": 60}


def repo_key(cwd: str = "") -> str:
    base = vcs.root(cwd) or cwd or str(Path.cwd())
    return hashlib.sha1(base.encode("utf-8", "replace")).hexdigest()[:12]


def path_for(cwd: str = "") -> Path:
    return ROOT / f"{repo_key(cwd)}.json"


def _load(cwd: str = "") -> List[Dict[str, Any]]:
    try:
        data = json.loads(path_for(cwd).read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(cwd: str, entries: List[Dict[str, Any]]) -> None:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        path = path_for(cwd)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries[-MAX_ENTRIES:], indent=2))
        tmp.replace(path)
    except Exception:
        pass


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _prune(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep recent general notes, but cap noisy self-learning kinds so the file
    # stays small and the prompt stays useful.
    kept: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for entry in reversed(entries):
        kind = str(entry.get("kind") or "note")
        limit = KIND_LIMITS.get(kind)
        if limit is not None:
            n = counts.get(kind, 0)
            if n >= limit:
                continue
            counts[kind] = n + 1
        kept.append(entry)
    kept.reverse()
    return kept[-MAX_ENTRIES:]


def remember(cwd: str, kind: str, text: str, meta: Any = None) -> Dict[str, Any]:
    text = re.sub(r"\s+", " ", (text or "").strip())[:2000]
    if not text:
        return {}
    kind = kind or "note"
    entries = _load(cwd)
    key = _norm(text)
    # Deduplicate near-identical lessons/patterns and strengthen them instead of
    # appending noise.
    for entry in reversed(entries[-120:]):
        if str(entry.get("kind") or "") != kind:
            continue
        existing = _norm(str(entry.get("text") or ""))
        if not existing:
            continue
        if existing == key or (len(key) > 40 and (key in existing or existing in key)):
            entry["ts"] = time.time()
            entry["count"] = int(entry.get("count") or 1) + 1
            if meta and isinstance(meta, dict) and isinstance(entry.get("meta"), dict):
                entry["meta"].update(meta)
            elif meta and entry.get("meta") is None:
                entry["meta"] = meta
            _save(cwd, _prune(entries))
            return entry
    entry = {
        "ts": time.time(),
        "kind": kind,
        "text": text,
        "count": 1,
        "meta": meta,
    }
    entries.append(entry)
    _save(cwd, _prune(entries))
    return entry


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_][\w-]*", (text or "").lower()) if w not in STOP and len(w) > 2}


def recall(cwd: str = "", query: str = "", k: int = 5) -> List[Dict[str, Any]]:
    entries = _load(cwd)
    if not entries:
        return []
    q = _tokens(query)
    scored: List[tuple[int, Dict[str, Any]]] = []
    for entry in entries:
        words = _tokens(str(entry.get("text") or ""))
        score = len(q & words)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:max(0, k)]]


def recent(cwd: str = "", k: int = 5) -> List[Dict[str, Any]]:
    return _load(cwd)[-max(0, k):]


def format_memories(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = ["Project memory (learned earlier in this repo):"]
    for entry in entries:
        kind = str(entry.get("kind") or "note")
        lines.append(f"- [{kind}] {str(entry.get('text') or '').strip()[:400]}")
    return "\n".join(lines)


def by_kind(cwd: str = "", kinds: Optional[set] = None, k: int = 12) -> List[Dict[str, Any]]:
    """Most-reinforced entries of the requested kinds (self-learning)."""
    wanted = kinds if kinds is not None else LESSON_KINDS
    entries = [e for e in _load(cwd) if str(e.get("kind") or "") in wanted]
    entries.sort(key=lambda e: (int(e.get("count") or 1), float(e.get("ts") or 0)), reverse=True)
    return entries[:max(0, k)]


def format_lessons(entries: List[Dict[str, Any]]) -> str:
    """Compact 'do not repeat' block injected into the system prompt."""
    if not entries:
        return ""
    lines = ["Learned lessons (self-learning - apply before acting):"]
    for entry in entries:
        kind = str(entry.get("kind") or "lesson")
        text = str(entry.get("text") or "").strip()[:320]
        count = int(entry.get("count") or 1)
        suffix = f" (hit {count}x)" if count > 1 else ""
        lines.append(f"- [{kind}] {text}{suffix}")
    return "\n".join(lines)


def format_patterns(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = ["Observed patterns (reuse when the same signal appears):"]
    for entry in entries:
        text = str(entry.get("text") or "").strip()[:320]
        count = int(entry.get("count") or 1)
        suffix = f" (seen {count}x)" if count > 1 else ""
        lines.append(f"- {text}{suffix}")
    return "\n".join(lines)


def forget_kinds(cwd: str = "", kinds: Optional[set] = None) -> int:
    wanted = kinds if kinds is not None else LESSON_KINDS
    entries = _load(cwd)
    keep = [e for e in entries if str(e.get("kind") or "") not in wanted]
    removed = len(entries) - len(keep)
    if removed:
        _save(cwd, keep)
    return removed


def forget(cwd: str = "", kind: str = "") -> int:
    """Drop entries of one kind (or all lessons/patterns). Returns removed count."""
    entries = _load(cwd)
    if kind:
        keep = [e for e in entries if str(e.get("kind") or "") != kind]
    else:
        keep = [e for e in entries if str(e.get("kind") or "") not in LESSON_KINDS]
    removed = len(entries) - len(keep)
    if removed:
        _save(cwd, keep)
    return removed
