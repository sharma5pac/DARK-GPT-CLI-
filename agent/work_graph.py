"""Per-target exploration/work graph, linked to the shared world/asset graph.

``graph.json`` (knowledge_graph.py) answers "what exists".  This file answers
"what was tried, by whom, and why it counts".  Work nodes carry anchors to
world nodes and evidence refs, so coverage is computed from real state instead
of a decorative percentage.  The journal is append-only: nothing is deleted.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent import knowledge_graph as kg

VERSION = 1
DONE_STATUSES = {"done", "completed", "proved", "verified", "fixed", "resolved"}
FAILED_STATUSES = {"failed", "blocked", "abandoned", "error"}
COVERABLE_KINDS = ("host", "service", "endpoint", "finding")


def work_path(target: str) -> Path:
    return kg.target_dir(target) / "work_graph.json"


def _empty(target: str) -> Dict[str, Any]:
    now = time.time()
    return {
        "version": VERSION,
        "target": target,
        "slug": kg.slugify(target),
        "created_at": now,
        "updated_at": now,
        "nodes": {},
        "journal": [],
    }


def load(target: str) -> Dict[str, Any]:
    path = work_path(target)
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("nodes"), dict):
            base = _empty(target)
            base.update(data)
            base.setdefault("journal", [])
            return base
    except Exception:
        pass
    return _empty(target)


def save(target: str, graph: Dict[str, Any]) -> Dict[str, Any]:
    path = work_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph["updated_at"] = time.time()
    graph["journal"] = (graph.get("journal") or [])[-3000:]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return graph


def node_key(kind: str, identity: str) -> str:
    raw = f"work:{kind}:{(identity or '').strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _journal(graph: Dict[str, Any], op: str, **fields: Any) -> None:
    entry = {"ts": time.time(), "op": op}
    entry.update({k: v for k, v in fields.items() if v not in (None, "", [], {})})
    graph.setdefault("journal", []).append(entry)


def _merge_attrs(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    for key, value in (incoming or {}).items():
        if value in (None, "", [], {}, ()):
            continue
        if isinstance(value, list) and isinstance(merged.get(key), list):
            seen = list(merged[key])
            for item in value:
                if item not in seen:
                    seen.append(item)
            merged[key] = seen[-100:]
        else:
            merged[key] = value
    return merged


def upsert_node(
    target: str,
    kind: str,
    identity: str,
    attrs: Optional[Dict[str, Any]] = None,
    *,
    status: str = "open",
    owner: str = "",
    anchors: Optional[Iterable[Dict[str, Any]]] = None,
    source: str = "",
    note: str = "",
) -> Dict[str, Any]:
    graph = load(target)
    key = node_key(kind, identity)
    node = graph["nodes"].get(key) or {
        "key": key,
        "kind": kind,
        "id": identity,
        "attrs": {},
        "anchors": [],
        "created_at": time.time(),
    }
    node["attrs"] = _merge_attrs(node.get("attrs"), attrs)
    if status:
        node["status"] = status
    if owner:
        node["owner"] = owner
    seen = {tuple(sorted((a or {}).items())) for a in node.get("anchors", []) if isinstance(a, dict)}
    for anchor in anchors or []:
        if not isinstance(anchor, dict):
            continue
        sig = tuple(sorted(anchor.items()))
        if sig not in seen:
            node.setdefault("anchors", []).append(anchor)
            seen.add(sig)
    node["anchors"] = node.get("anchors", [])[-100:]
    if source and source not in node.get("sources", []):
        node.setdefault("sources", []).append(source)
        node["sources"] = node["sources"][-20:]
    node["updated_at"] = time.time()
    graph["nodes"][key] = node
    _journal(graph, "upsert", key=key, kind=kind, status=status, owner=owner, source=source, note=note)
    save(target, graph)
    return node


def set_status(target: str, key: str, status: str, *, owner: str = "", note: str = "", attrs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    graph = load(target)
    node = graph["nodes"].get(str(key))
    if not node:
        return None
    node["status"] = status
    if owner:
        node["owner"] = owner
    if attrs:
        node["attrs"] = _merge_attrs(node.get("attrs"), attrs)
    node["updated_at"] = time.time()
    _journal(graph, "status", key=str(key), status=status, owner=owner, note=note)
    save(target, graph)
    return node


def _world_key(target: str, kind: str, identity: str) -> str:
    needle = str(identity or "").strip().lower()
    if not needle:
        return ""
    for node in kg.query(target, kind=kind, text=identity, limit=20):
        if str(node.get("id") or "").strip().lower() == needle:
            return str(node.get("key") or "")
    return ""


def anchor(
    target: str,
    work_kind: str,
    work_id: str,
    world_kind: str,
    world_id: str,
    *,
    evidence_ref: str = "",
    finding_key: str = "",
    source: str = "anchor",
) -> Dict[str, Any]:
    """Link a work node to a real node in the world graph (no phantom assets)."""
    world_key = _world_key(target, world_kind, world_id)
    if not world_key:
        return {"ok": False, "error": f"world node not found: {world_kind}:{world_id}"}
    node = upsert_node(
        target, work_kind, work_id,
        anchors=[{
            "graph": "world",
            "node_key": world_key,
            "kind": world_kind,
            "id": world_id,
            "evidence_ref": evidence_ref,
            "finding_key": finding_key,
        }],
        source=source,
    )
    return {"ok": True, "work_key": node["key"], "world_key": world_key}


def query(target: str, *, kind: str = "", status: str = "", text: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    graph = load(target)
    needle = str(text or "").strip().lower()
    out = []
    for node in reversed(list(graph["nodes"].values())):
        if kind and str(node.get("kind")) != kind:
            continue
        if status and str(node.get("status")) != status:
            continue
        if needle and needle not in f"{node.get('id')} {json.dumps(node.get('attrs') or {})}".lower():
            continue
        out.append(node)
        if len(out) >= max(1, min(limit, 500)):
            break
    return out


def coverage(target: str, goal_id: str = "") -> Dict[str, Any]:
    """Coverage is derived from completed work anchors that point to live world nodes."""
    work = load(target)
    world = kg._load(target)
    world_nodes = {
        key: node for key, node in (world.get("nodes") or {}).items()
        if node.get("kind") in COVERABLE_KINDS
    }
    touched: Dict[str, str] = {}
    claimed: Dict[str, str] = {}
    for node in work["nodes"].values():
        for anchor in node.get("anchors") or []:
            key = str((anchor or {}).get("node_key") or "")
            if key not in world_nodes:
                continue
            claimed.setdefault(key, node.get("id") or node.get("key") or "")
            if node.get("status") in DONE_STATUSES:
                touched.setdefault(key, node.get("id") or node.get("key") or "")
    per_kind: Dict[str, Dict[str, int]] = {}
    uncovered: List[Dict[str, Any]] = []
    for kind in COVERABLE_KINDS:
        keys = [k for k, n in world_nodes.items() if n.get("kind") == kind]
        covered = [k for k in keys if k in touched]
        per_kind[kind] = {"total": len(keys), "covered": len(covered)}
        for key in keys:
            if key in touched:
                continue
            item = {"kind": kind, "key": key, "id": world_nodes[key].get("id"), "claimed_by": claimed.get(key, "")}
            if not goal_id:
                uncovered.append(item)
            else:
                # A goal only owns assets named in its scope, or all assets when it has none.
                gnode = work["nodes"].get(node_key("goal", goal_id))
                scope = ((gnode or {}).get("attrs") or {}).get("scope") or []
                if not scope or any(str(s).lower() in str(item["id"]).lower() for s in scope):
                    uncovered.append(item)
    total = sum(v["total"] for v in per_kind.values())
    covered_total = sum(v["covered"] for v in per_kind.values())
    return {
        "target": target,
        "goal_id": goal_id,
        "world_nodes": len(world_nodes),
        "covered": covered_total,
        "uncovered": uncovered,
        "claims": claimed,
        "touched": touched,
        "per_kind": per_kind,
        "percent": round(100.0 * covered_total / total, 1) if total else 0.0,
    }


def summary(target: str, limit: int = 8) -> Dict[str, Any]:
    graph = load(target)
    by_kind: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    for node in graph["nodes"].values():
        by_kind[str(node.get("kind"))] = by_kind.get(str(node.get("kind")), 0) + 1
        by_status[str(node.get("status"))] = by_status.get(str(node.get("status")), 0) + 1
    cov = coverage(target)
    return {
        "target": target,
        "path": str(work_path(target)),
        "nodes": len(graph["nodes"]),
        "by_kind": by_kind,
        "by_status": by_status,
        "coverage": {"percent": cov["percent"], "covered": cov["covered"], "world_nodes": cov["world_nodes"]},
        "recent": list(reversed(graph.get("journal", [])))[:limit],
    }


def export(target: str, fmt: str = "json") -> Dict[str, Any]:
    graph = load(target)
    fmt = str(fmt or "json").lower()
    if fmt == "json":
        return {"ok": True, "format": "json", "graph": graph}
    lines = ["digraph work_graph {", "  rankdir=LR;"]
    for key, node in graph["nodes"].items():
        label = f"{node.get('kind')}:{str(node.get('id') or '')[:40]}".replace('"', "'")
        lines.append(f'  "{key}" [label="{label}"];')
        for anchor in node.get("anchors") or []:
            dst = str((anchor or {}).get("node_key") or "")
            if dst:
                lines.append(f'  "{key}" -> "world_{dst}" [label="anchor"];')
    lines.append("}")
    return {"ok": True, "format": "dot", "dot": "\n".join(lines)}
