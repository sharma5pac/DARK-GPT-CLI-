"""Per-target planner/task graph for adaptive pentest loops.

Persists objective, hypotheses, tasks, observations, and dead-ends. Used by the
agent to maintain state across turns and avoid repeating failed routes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import knowledge_graph as kg


def planner_path(target: str) -> Path:
    return kg.target_dir(target) / "planner.json"


def _empty(target: str) -> Dict[str, Any]:
    return {
        "version": 1,
        "target": target,
        "objective": "",
        "hypotheses": [],  # {id, text, status, confidence, evidence_ref, updated_at}
        "tasks": [],       # {id, title, status, owner, notes, updated_at}
        "observations": [],# {ts, tool, observed, hypothesis_delta, next_action}
        "dead_ends": [],   # {signature, reason, pivot, count, last_ts}
        "updated_at": time.time(),
    }


def load(target: str) -> Dict[str, Any]:
    p = planner_path(target)
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            base = _empty(target)
            base.update(data)
            base.setdefault("hypotheses", [])
            base.setdefault("tasks", [])
            base.setdefault("observations", [])
            base.setdefault("dead_ends", [])
            return base
    except Exception:
        pass
    return _empty(target)


def save(target: str, data: Dict[str, Any]) -> Dict[str, Any]:
    p = planner_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(p)
    return data


def _id(prefix: str, items: List[Dict[str, Any]]) -> str:
    return f"{prefix}-{len(items)+1}-{int(time.time())%100000}"


def set_objective(target: str, objective: str) -> Dict[str, Any]:
    d = load(target)
    d["objective"] = str(objective or "").strip()[:1000]
    return save(target, d)


def add_hypothesis(target: str, text: str, status: str = "open", confidence: Any = None, evidence_ref: str = "") -> Dict[str, Any]:
    d = load(target)
    text = str(text or "").strip()
    if not text:
        return d
    for h in d["hypotheses"]:
        if str(h.get("text", "")).strip().lower() == text.lower():
            h["status"] = status or h.get("status", "open")
            if confidence is not None: h["confidence"] = confidence
            if evidence_ref: h["evidence_ref"] = evidence_ref
            h["updated_at"] = time.time()
            return save(target, d)
    d["hypotheses"].append({
        "id": _id("h", d["hypotheses"]), "text": text[:500], "status": status or "open",
        "confidence": confidence, "evidence_ref": evidence_ref, "updated_at": time.time(),
    })
    d["hypotheses"] = d["hypotheses"][-100:]
    return save(target, d)


def add_task(target: str, title: str, status: str = "todo", owner: str = "", notes: str = "") -> Dict[str, Any]:
    d = load(target)
    title = str(title or "").strip()
    if not title:
        return d
    for t in d["tasks"]:
        if str(t.get("title", "")).strip().lower() == title.lower():
            t["status"] = status or t.get("status", "todo")
            if owner: t["owner"] = owner
            if notes: t["notes"] = notes
            t["updated_at"] = time.time()
            return save(target, d)
    d["tasks"].append({
        "id": _id("t", d["tasks"]), "title": title[:300], "status": status or "todo",
        "owner": owner, "notes": notes[:1000], "updated_at": time.time(),
    })
    d["tasks"] = d["tasks"][-200:]
    return save(target, d)


def record_observation(target: str, *, tool: str = "", observed: str = "", hypothesis_delta: str = "", next_action: str = "") -> Dict[str, Any]:
    d = load(target)
    d["observations"].append({
        "ts": time.time(), "tool": str(tool or "")[:80],
        "observed": str(observed or "")[:800],
        "hypothesis_delta": str(hypothesis_delta or "")[:500],
        "next_action": str(next_action or "")[:500],
    })
    d["observations"] = d["observations"][-200:]
    return save(target, d)


def record_dead_end(target: str, signature: str, reason: str = "", pivot: str = "") -> Dict[str, Any]:
    d = load(target)
    sig = str(signature or "").strip().lower()[:200]
    if not sig:
        return d
    for de in d["dead_ends"]:
        if de.get("signature") == sig:
            de["count"] = int(de.get("count", 1)) + 1
            if reason: de["reason"] = str(reason)[:500]
            if pivot: de["pivot"] = str(pivot)[:500]
            de["last_ts"] = time.time()
            return save(target, d)
    d["dead_ends"].append({"signature": sig, "reason": str(reason)[:500], "pivot": str(pivot)[:500], "count": 1, "last_ts": time.time()})
    d["dead_ends"] = d["dead_ends"][-100:]
    return save(target, d)


def summary(target: str) -> Dict[str, Any]:
    d = load(target)
    open_h = [h for h in d.get("hypotheses", []) if h.get("status") in ("open", "active", "testing")]
    open_t = [t for t in d.get("tasks", []) if t.get("status") in ("todo", "doing", "active")]
    return {
        "target": target,
        "objective": d.get("objective", ""),
        "open_hypotheses": open_h[-5:],
        "active_tasks": open_t[-5:],
        "recent_observations": d.get("observations", [])[-5:],
        "dead_ends": d.get("dead_ends", [])[-5:],
        "counts": {
            "hypotheses": len(d.get("hypotheses", [])),
            "tasks": len(d.get("tasks", [])),
            "observations": len(d.get("observations", [])),
            "dead_ends": len(d.get("dead_ends", [])),
        },
        "path": str(planner_path(target)),
    }
