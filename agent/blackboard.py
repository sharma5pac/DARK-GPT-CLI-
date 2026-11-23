"""Shared blackboard for multi-agent coordination.

Stores Facts, Intents, and Human Hints per target. Workers can post/read without
central bottlenecking. Simple JSON file with append/update semantics.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import knowledge_graph as kg


def blackboard_path(target: str) -> Path:
    return kg.target_dir(target) / "blackboard.json"


def _empty(target: str) -> Dict[str, Any]:
    return {"version": 1, "target": target, "updated_at": time.time(), "items": []}


def load(target: str) -> Dict[str, Any]:
    p = blackboard_path(target)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data
    except Exception:
        pass
    return _empty(target)


def save(target: str, data: Dict[str, Any]) -> Dict[str, Any]:
    p = blackboard_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return data


def post(target: str, kind: str, text: str, *, status: str = "open", source: str = "agent", refs: Optional[List[str]] = None, confidence: Any = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kind = str(kind or "intent").lower().strip()
    if kind not in {"fact", "intent", "hint"}:
        kind = "intent"
    d = load(target)
    item = {
        "id": "bb_" + uuid.uuid4().hex[:12],
        "kind": kind,
        "text": str(text or "").strip()[:2000],
        "status": status,
        "source": source,
        "refs": list(refs or [])[:20],
        "confidence": confidence,
        "meta": meta or {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    d["items"].append(item)
    d["items"] = d["items"][-1000:]
    save(target, d)
    return item


def list_items(target: str, kind: str = "", status: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    d = load(target)
    out = []
    for it in reversed(d.get("items") or []):
        if kind and it.get("kind") != kind:
            continue
        if status and it.get("status") != status:
            continue
        out.append(it)
        if len(out) >= max(0, limit):
            break
    return out


def claim(target: str, item_id: str, owner: str) -> Dict[str, Any]:
    d = load(target)
    for it in d.get("items") or []:
        if it.get("id") == item_id:
            it["status"] = "claimed"
            it["owner"] = owner
            it["updated_at"] = time.time()
            save(target, d)
            return {"ok": True, "item": it}
    return {"ok": False, "error": "item not found"}


def resolve(target: str, item_id: str, status: str = "done", note: str = "") -> Dict[str, Any]:
    d = load(target)
    for it in d.get("items") or []:
        if it.get("id") == item_id:
            it["status"] = status
            if note:
                it.setdefault("meta", {})["resolution_note"] = note[:500]
            it["updated_at"] = time.time()
            save(target, d)
            return {"ok": True, "item": it}
    return {"ok": False, "error": "item not found"}


def summary(target: str) -> Dict[str, Any]:
    d = load(target)
    counts = {}
    for it in d.get("items") or []:
        k = f"{it.get('kind')}:{it.get('status')}"
        counts[k] = counts.get(k, 0) + 1
    return {"target": target, "path": str(blackboard_path(target)), "count": len(d.get("items") or []), "counts": counts, "recent": list_items(target, limit=10)}
