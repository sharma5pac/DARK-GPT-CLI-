"""Append-only evidence ledger for pentest/recon runs.

Every tool result can be recorded with timestamp, args summary, redacted output
preview, and SHA-256 of the full result. Evidence is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
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


def _entry_id(ts: float, tool: str, result_sha256: str) -> str:
    raw = f"{ts}|{tool}|{result_sha256}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(e, dict):
        return {}
    if not e.get("id"):
        try:
            e["id"] = _entry_id(float(e.get("ts") or 0), str(e.get("tool") or ""), str(e.get("result_sha256") or ""))
        except Exception:
            e["id"] = _entry_id(0.0, str(e.get("tool") or ""), str(e.get("result_sha256") or ""))
    return e


def record(
    target: str,
    *,
    tool: str,
    args: Any = None,
    result: str = "",
    ok: bool = True,
    source: str = "agent",
    tags: Optional[List[str]] = None,
    artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = evidence_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    ts = time.time()
    digest = sha256_text(result_text)
    entry = {
        "id": _entry_id(ts, tool, digest),
        "ts": ts,
        "target": target,
        "tool": tool,
        "ok": bool(ok),
        "source": source,
        "tags": list(tags or []),
        "args": _redact(args),
        "result_preview": result_text[:6000],
        "result_sha256": digest,
        "result_bytes": len(result_text.encode("utf-8", "replace")),
    }
    if isinstance(artifact, dict) and artifact:
        entry["artifact"] = _redact(artifact)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_entries(target: str, limit: int = 200) -> List[Dict[str, Any]]:
    path = evidence_path(target)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-max(0, limit):]:
        try:
            out.append(_normalize_entry(json.loads(line)))
        except Exception:
            continue
    return out


def has_ref(target: str, ref: str) -> bool:
    """Check if an evidence ref exists. Accepts entry id or sha256 prefix/full hash."""
    ref = str(ref or "").strip().lower()
    if not ref:
        return False
    for e in read_entries(target, limit=100000):
        eid = str(e.get("id") or "").lower()
        sha = str(e.get("result_sha256") or "").lower()
        if ref == eid or ref == sha or sha.startswith(ref) or eid.startswith(ref):
            return True
    return False


def get_by_ref(target: str, ref: str) -> Optional[Dict[str, Any]]:
    ref = str(ref or "").strip().lower()
    if not ref:
        return None
    for e in read_entries(target, limit=100000):
        eid = str(e.get("id") or "").lower()
        sha = str(e.get("result_sha256") or "").lower()
        if ref == eid or ref == sha or sha.startswith(ref) or eid.startswith(ref):
            return e
    return None




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

def save_repro_artifact(target: str, finding_id: str, content: str, kind: str = "py") -> Dict[str, Any]:
    """Save a deterministic replay script for a finding. Never overwrites silently; versioned."""
    from agent import knowledge_graph as kg
    d = kg.target_dir(target) / "repro"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", finding_id)[:80] or "finding"
    ext = ".py" if kind == "py" else ".sh"
    base = d / f"{safe}{ext}"
    path = base
    i = 2
    while path.exists():
        path = d / f"{safe}_v{i}{ext}"
        i += 1
    path.write_text(content, encoding="utf-8")
    try:
        os.chmod(path, 0o700)
    except Exception:
        pass
    return {"ok": True, "path": str(path), "sha256": sha256_text(content), "bytes": len(content.encode("utf-8", "replace"))}
