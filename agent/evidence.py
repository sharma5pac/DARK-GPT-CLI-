"""Append-only evidence ledger for pentest/recon runs.

Every tool result can be recorded with timestamp, args summary, redacted output
preview, and SHA-256 of the full result. Evidence is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import knowledge_graph as kg

SENSITIVE_KEYS = {"authorization", "cookie", "token", "api_key", "apikey", "password", "secret", "key"}


def evidence_path(target: str) -> Path:
    return kg.target_dir(target) / "evidence.jsonl"


def _redact(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[max-depth]"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(s in lk for s in SENSITIVE_KEYS):
                out[k] = "[REDACTED]"
            else:
                out[k] = _redact(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_redact(v, depth + 1) for v in obj[:50]]
    if isinstance(obj, str) and len(obj) > 4000:
        return obj[:4000] + f"...[truncated {len(obj)-4000}]"
    return obj


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


def record(
    target: str,
    *,
    tool: str,
    args: Any = None,
    result: str = "",
    ok: bool = True,
    source: str = "agent",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    path = evidence_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    entry = {
        "ts": time.time(),
        "target": target,
        "tool": tool,
        "ok": bool(ok),
        "source": source,
        "tags": list(tags or []),
        "args": _redact(args),
        "result_preview": result_text[:6000],
        "result_sha256": sha256_text(result_text),
        "result_bytes": len(result_text.encode("utf-8", "replace")),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_entries(target: str, limit: int = 200) -> List[Dict[str, Any]]:
    path = evidence_path(target)
    if not path.exists():
        return []
    # Read tail efficiently enough for typical ledgers; keep simple.
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out = []
    for line in lines[-max(0, limit):]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def summary(target: str) -> Dict[str, Any]:
    entries = read_entries(target, limit=100000)
    by_tool: Dict[str, int] = {}
    ok_count = 0
    for e in entries:
        t = str(e.get("tool") or "unknown")
        by_tool[t] = by_tool.get(t, 0) + 1
        if e.get("ok"):
            ok_count += 1
    return {
        "target": target,
        "path": str(evidence_path(target)),
        "count": len(entries),
        "ok": ok_count,
        "failed": len(entries) - ok_count,
        "by_tool": dict(sorted(by_tool.items(), key=lambda kv: kv[1], reverse=True)[:30]),
        "last_ts": entries[-1]["ts"] if entries else None,
    }