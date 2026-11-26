"""Per-target knowledge graph for recon/pentest work.

Persists hosts -> services -> endpoints -> findings so recon can resume and
patterns/evidence transfer across engagements. Data is never deleted implicitly;
merges are additive and idempotent by node key.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path.home() / ".config" / "darkgpt" / "targets"


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip().lower()).strip("-")
    text = text[:80] or "target"
    return text


def target_dir(target: str) -> Path:
    return ROOT / slugify(target)


def graph_path(target: str) -> Path:
    return target_dir(target) / "graph.json"


def _now() -> float:
    return time.time()


def _empty(target: str) -> Dict[str, Any]:
    return {
        "version": 1,
        "target": target,
        "slug": slugify(target),
        "created_at": _now(),
        "updated_at": _now(),
        "nodes": {},  # key -> node
        "edges": [],  # {src, dst, rel, ts}
    }


def _load(target: str) -> Dict[str, Any]:
    path = graph_path(target)
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("nodes"), dict):
            data.setdefault("edges", [])
            data.setdefault("target", target)
            data.setdefault("slug", slugify(target))
            return data
    except Exception:
        pass
    return _empty(target)


def _save(target: str, graph: Dict[str, Any]) -> None:
    path = graph_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph["updated_at"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _node_key(kind: str, identity: str) -> str:
    raw = f"{kind}:{(identity or '').strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def upsert_node(
    target: str,
    kind: str,
    identity: str,
    attrs: Optional[Dict[str, Any]] = None,
    *,
    source: str = "",
) -> Dict[str, Any]:
    graph = _load(target)
    key = _node_key(kind, identity)
    node = graph["nodes"].get(key) or {
        "key": key,
        "kind": kind,
        "id": identity,
        "attrs": {},
        "sources": [],
        "first_seen": _now(),
    }
    node["last_seen"] = _now()
    merged = dict(node.get("attrs") or {})
    for k, v in (attrs or {}).items():
        if v in (None, "", [], {}, ()):  # don't clobber with empties
            continue
        if isinstance(v, list) and isinstance(merged.get(k), list):
            existing = list(merged[k])
            for item in v:
                if item not in existing:
                    existing.append(item)
            merged[k] = existing[-50:]
        else:
            merged[k] = v
    node["attrs"] = merged
    if source and source not in node.get("sources", []):
        node.setdefault("sources", []).append(source)
        node["sources"] = node["sources"][-20:]
    graph["nodes"][key] = node
    _save(target, graph)
    return node


def update_node_attrs(target: str, key: str, attrs: Dict[str, Any], source: str = "") -> Optional[Dict[str, Any]]:
    """Update an existing node by stable key (never create a duplicate node)."""
    graph = _load(target)
    node = graph["nodes"].get(str(key))
    if not node:
        return None
    merged = dict(node.get("attrs") or {})
    for k, v in (attrs or {}).items():
        if v in (None, "", [], {}, ()):
            continue
        if isinstance(v, list) and isinstance(merged.get(k), list):
            for item in v:
                if item not in merged[k]:
                    merged[k].append(item)
            merged[k] = merged[k][-50:]
        else:
            merged[k] = v
    node["attrs"] = merged
    node["last_seen"] = _now()
    if source and source not in node.get("sources", []):
        node.setdefault("sources", []).append(source)
        node["sources"] = node["sources"][-20:]
    graph["nodes"][str(key)] = node
    _save(target, graph)
    return node


def add_edge(target: str, src_key: str, dst_key: str, rel: str, *, created_by: str = "", evidence_id: str = "", confidence: Any = None, meta: Optional[Dict[str, Any]] = None) -> None:
    graph = _load(target)
    if src_key not in graph["nodes"] or dst_key not in graph["nodes"]:
        return
    for e in graph.get("edges", []):
        if e.get("src") == src_key and e.get("dst") == dst_key and e.get("rel") == rel:
            # enrich provenance without duplicating edge
            if created_by and created_by not in (e.get("created_by") or []):
                e.setdefault("created_by", []).append(created_by)
                e["created_by"] = e["created_by"][-10:]
            if evidence_id and evidence_id not in (e.get("evidence_ids") or []):
                e.setdefault("evidence_ids", []).append(evidence_id)
                e["evidence_ids"] = e["evidence_ids"][-20:]
            if confidence is not None:
                e["confidence"] = confidence
            if meta:
                e.setdefault("meta", {}).update({k: v for k, v in meta.items() if v not in (None, "", [], {})})
            _save(target, graph)
            return
    edge = {"src": src_key, "dst": dst_key, "rel": rel, "ts": _now()}
    if created_by: edge["created_by"] = [created_by]
    if evidence_id: edge["evidence_ids"] = [evidence_id]
    if confidence is not None: edge["confidence"] = confidence
    if meta: edge["meta"] = dict(meta)
    graph.setdefault("edges", []).append(edge)
    graph["edges"] = graph["edges"][-4000:]
    _save(target, graph)


def add_host(target: str, host: str, attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    return upsert_node(target, "host", host, {"host": host, **(attrs or {})}, source=source)


def add_service(target: str, host: str, port: Any, proto: str = "tcp", attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    identity = f"{host}:{port}/{proto}"
    svc = upsert_node(target, "service", identity, {"host": host, "port": int(port), "proto": proto, **(attrs or {})}, source=source)
    h = add_host(target, host, source=source)
    add_edge(target, h["key"], svc["key"], "has_service", created_by=source or "kg")
    return svc


def add_endpoint(target: str, url: str, attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    ep = upsert_node(target, "endpoint", url, {"url": url, **(attrs or {})}, source=source)
    # link to host if parseable
    m = re.match(r"https?://([^/:]+)", url or "", re.I)
    if m:
        h = add_host(target, m.group(1).lower(), source=source)
        add_edge(target, h["key"], ep["key"], "has_endpoint")
    return ep


FINDING_STATUSES = {"attempted", "proved", "false_positive"}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _normalize_status(status: Any) -> str:
    s = str(status or "attempted").strip().lower().replace(" ", "_")
    aliases = {"verified": "proved", "confirmed": "proved", "unverified": "attempted", "fp": "false_positive", "false-positive": "false_positive"}
    s = aliases.get(s, s)
    return s if s in FINDING_STATUSES else "attempted"


def _evidence_ref_exists(target: str, evidence_ref: str) -> bool:
    ref = str(evidence_ref or "").strip()
    if not ref:
        return False
    try:
        from agent import evidence as _ev
        return _ev.has_ref(target, ref)
    except Exception:
        # fallback: direct scan
        try:
            p = target_dir(target) / "evidence.jsonl"
            if not p.exists():
                return False
            needle = ref.lower()
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                eid = str(e.get("id") or "").lower()
                sha = str(e.get("result_sha256") or "").lower()
                if needle in (eid, sha) or sha.startswith(needle) or eid.startswith(needle):
                    return True
        except Exception:
            return False
    return False


def add_credential(target: str, name: str, kind: str = "token", attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    # Never store raw secrets; store redacted ref/hash only.
    a = dict(attrs or {})
    secret = str(a.pop("secret", "") or a.pop("value", "") or "")
    ref = a.get("ref") or (hashlib.sha256(secret.encode("utf-8", "replace")).hexdigest()[:16] if secret else "")
    a.update({"name": name, "kind": kind, "ref": ref, "redacted": bool(secret) or bool(a.get("redacted"))})
    node = upsert_node(target, "credential", f"{kind}:{name}:{ref}", a, source=source)
    return node


def add_auth_state(target: str, host: str, state: str, attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    identity = f"{host}:{state}"
    node = upsert_node(target, "auth_state", identity, {"host": host, "state": state, **(attrs or {})}, source=source)
    h = add_host(target, host, source=source)
    add_edge(target, h["key"], node["key"], "has_auth_state", created_by=source or "kg")
    return node


def add_param(target: str, endpoint: str, name: str, location: str = "query", attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    identity = f"{endpoint}:{location}:{name}"
    node = upsert_node(target, "param", identity, {"endpoint": endpoint, "name": name, "location": location, **(attrs or {})}, source=source)
    ep = add_endpoint(target, endpoint, source=source)
    add_edge(target, ep["key"], node["key"], "has_param", created_by=source or "kg")
    return node


def add_fingerprint(target: str, asset: str, fingerprint: str, attrs: Optional[Dict[str, Any]] = None, source: str = "") -> Dict[str, Any]:
    identity = f"{asset}:{fingerprint[:80]}"
    node = upsert_node(target, "fingerprint", identity, {"asset": asset, "fingerprint": fingerprint[:500], **(attrs or {})}, source=source)
    return node


def add_finding(
    target: str,
    title: str,
    severity: str = "info",
    details: str = "",
    evidence_ref: str = "",
    attrs: Optional[Dict[str, Any]] = None,
    source: str = "",
) -> Dict[str, Any]:
    a = dict(attrs or {})
    status = _normalize_status(a.get("status", "attempted"))
    sev = str(severity or a.get("severity") or "info").strip().lower()
    if sev not in SEVERITY_ORDER:
        sev = "info"
    evidence_ref = str(evidence_ref or a.get("evidence_ref") or "").strip()
    oracle_ref = str(a.get("oracle_ref") or "").strip()
    oracle_verified = bool(a.get("oracle_verified")) or str(a.get("oracle_verdict") or "").lower() in {"different","reflected","potential_delay","confirmed","verified"}
    if status == "proved" and not _evidence_ref_exists(target, evidence_ref) and not (oracle_ref and oracle_verified):
        # proof gate: proved requires evidence ref or verified machine oracle
        a["proof_gate_blocked"] = True
        a["proof_gate_reason"] = "proved status requires evidence_ref in ledger or verified oracle_ref+oracle_verified"
        status = "attempted"
    a["status"] = status
    a["severity"] = sev
    identity = f"{sev}:{title}:{str(a.get('asset') or details)[:80]}"
    node = upsert_node(target, "finding", identity, {
        "title": title,
        "severity": sev,
        "status": status,
        "details": details[:4000],
        "evidence_ref": evidence_ref,
        "asset": str(a.get("asset") or ""),
        "repro": str(a.get("repro") or a.get("reproduction") or "")[:4000],
        "impact": str(a.get("impact") or "")[:2000],
        "remediation": str(a.get("remediation") or "")[:2000],
        "confidence": a.get("confidence"),
        **{k: v for k, v in a.items() if k not in {"title", "severity", "status", "details", "evidence_ref"}},
    }, source=source)
    return node


def link(target: str, src_kind: str, src_id: str, dst_kind: str, dst_id: str, rel: str) -> bool:
    graph = _load(target)
    sk = _node_key(src_kind, src_id)
    dk = _node_key(dst_kind, dst_id)
    if sk not in graph["nodes"] or dk not in graph["nodes"]:
        return False
    add_edge(target, sk, dk, rel)
    return True


def query(target: str, kind: str = "", text: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    graph = _load(target)
    needle = (text or "").lower().strip()
    out: List[Dict[str, Any]] = []
    for node in graph.get("nodes", {}).values():
        if kind and node.get("kind") != kind:
            continue
        if needle:
            hay = json.dumps(node, ensure_ascii=False).lower()
            if needle not in hay:
                continue
        out.append(node)
    out.sort(key=lambda n: float(n.get("last_seen") or 0), reverse=True)
    return out[:max(0, limit)]


def summary(target: str) -> Dict[str, Any]:
    graph = _load(target)
    counts: Dict[str, int] = {}
    for node in graph.get("nodes", {}).values():
        k = str(node.get("kind") or "unknown")
        counts[k] = counts.get(k, 0) + 1
    return {
        "target": graph.get("target", target),
        "slug": graph.get("slug", slugify(target)),
        "path": str(graph_path(target)),
        "updated_at": graph.get("updated_at"),
        "counts": counts,
        "edges": len(graph.get("edges") or []),
        "recent_findings": [
            {
                "severity": n.get("attrs", {}).get("severity"),
                "status": n.get("attrs", {}).get("status", "attempted"),
                "title": n.get("attrs", {}).get("title"),
                "details": (n.get("attrs", {}).get("details") or "")[:160],
                "evidence_ref": n.get("attrs", {}).get("evidence_ref", ""),
            }
            for n in query(target, kind="finding", limit=10)
        ],
    }


def list_targets(limit: int = 50) -> List[Dict[str, Any]]:
    if not ROOT.exists():
        return []
    items = []
    for path in ROOT.glob("*/graph.json"):
        try:
            data = json.loads(path.read_text())
            nodes = data.get("nodes") or {}
            counts: Dict[str, int] = {}
            for n in nodes.values():
                k = str(n.get("kind") or "unknown")
                counts[k] = counts.get(k, 0) + 1
            items.append({
                "target": data.get("target") or path.parent.name,
                "slug": data.get("slug") or path.parent.name,
                "updated_at": data.get("updated_at") or 0,
                "counts": counts,
                "path": str(path),
            })
        except Exception:
            continue
    items.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    return items[:max(0, limit)]


def merge(target: str, other_graph: Dict[str, Any], source: str = "merge") -> Dict[str, Any]:
    """Additively merge another graph/export into target. Never deletes."""
    graph = _load(target)
    key_map: Dict[str, str] = {}
    for node in (other_graph.get("nodes") or {}).values():
        kind = str(node.get("kind") or "unknown")
        identity = str(node.get("id") or node.get("key") or "")
        if not identity:
            continue
        new_node = upsert_node(target, kind, identity, node.get("attrs") or {}, source=source)
        key_map[str(node.get("key") or "")] = new_node["key"]
    graph = _load(target)
    for edge in other_graph.get("edges") or []:
        src = key_map.get(str(edge.get("src") or ""))
        dst = key_map.get(str(edge.get("dst") or ""))
        rel = str(edge.get("rel") or "related")
        if src and dst:
            existing = {(e.get("src"), e.get("dst"), e.get("rel")) for e in graph.get("edges", [])}
            if (src, dst, rel) not in existing:
                graph.setdefault("edges", []).append({"src": src, "dst": dst, "rel": rel, "ts": _now()})
    graph["edges"] = (graph.get("edges") or [])[-2000:]
    _save(target, graph)
    return summary(target)


def export_graph(target: str, fmt: str = "json") -> Dict[str, Any]:
    fmt = str(fmt or "json").lower().strip()
    graph = _load(target)
    if fmt in ("json", ""):
        return {"format": "json", "graph": graph, "path": str(graph_path(target))}
    if fmt in ("dot", "graphviz"):
        lines = ["digraph KG {"]
        for k, n in (graph.get("nodes") or {}).items():
            label = f"{n.get('kind')}\\n{str(n.get('id',''))[:40]}"
            label = label.replace('"', "'")
            lines.append(f'  "{k}" [label="{label}"];')
        for e in graph.get("edges") or []:
            rel = str(e.get("rel", "related")).replace('"', "'")
            lines.append(f'  "{e.get("src")}" -> "{e.get("dst")}" [label="{rel}"];')
        lines.append("}")
        dot = "\n".join(lines)
        out = target_dir(target) / "graph.dot"
        out.write_text(dot, encoding="utf-8")
        return {"format": "dot", "path": str(out), "bytes": len(dot.encode("utf-8"))}
    raise ValueError("format must be json or dot/graphviz")


# ----------------------------------------------------------------- attack paths
HIGH_VALUE_KINDS = {"credential", "auth_state", "finding"}
HIGH_VALUE_TERMS = ("admin", "root", "token", "secret", "aws", "s3", "bucket", "db", "database", "internal", "prod", "backup")


def _node_label(n: Dict[str, Any]) -> str:
    a = n.get("attrs") or {}
    return str(a.get("title") or a.get("url") or a.get("host") or a.get("name") or n.get("id") or n.get("key"))


def _is_high_value(n: Dict[str, Any]) -> bool:
    kind = str(n.get("kind") or "")
    if kind in HIGH_VALUE_KINDS:
        sev = str((n.get("attrs") or {}).get("severity") or "").lower()
        if kind == "finding" and sev not in ("critical", "high", "medium"):
            pass
        else:
            return True
    hay = json.dumps(n.get("attrs") or {}, ensure_ascii=False).lower()
    return any(t in hay for t in HIGH_VALUE_TERMS)


def rank_attack_paths(target: str, max_paths: int = 10, max_depth: int = 5) -> List[Dict[str, Any]]:
    """Rank simple multi-hop paths toward high-value nodes by heuristic impact."""
    graph = _load(target)
    nodes = graph.get("nodes") or {}
    edges = graph.get("edges") or []
    adj: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        adj.setdefault(str(e.get("src")), []).append(e)
    sources = [k for k, n in nodes.items() if str(n.get("kind")) in ("host", "endpoint", "service", "finding")]
    high = {k for k, n in nodes.items() if _is_high_value(n)}
    paths: List[Dict[str, Any]] = []
    for src in sources[:120]:
        stack = [(src, [src], [])]
        seen = set([src])
        while stack:
            cur, node_path, edge_path = stack.pop()
            if len(node_path) > 1 and cur in high:
                score = 0.0
                for nk in node_path:
                    n = nodes.get(nk, {})
                    kind = str(n.get("kind"))
                    a = n.get("attrs") or {}
                    sev = str(a.get("severity") or "").lower()
                    score += {"finding": 3, "credential": 4, "auth_state": 3, "endpoint": 1.5, "service": 1, "host": 0.5, "param": 1, "sink": 2}.get(kind, 0.5)
                    score += {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(sev, 0)
                    if _is_high_value(n): score += 2
                score += max(0, 2 - 0.2 * len(node_path))
                paths.append({
                    "score": round(score, 2),
                    "nodes": [ {"key": k, "kind": nodes.get(k,{}).get("kind"), "label": _node_label(nodes.get(k,{})), "id": nodes.get(k,{}).get("id")} for k in node_path ],
                    "edges": edge_path[-20:],
                })
                continue
            if len(node_path) >= max_depth:
                continue
            for e in adj.get(cur, [])[:30]:
                dst = str(e.get("dst"))
                if dst in node_path: continue
                stack.append((dst, node_path + [dst], edge_path + [{"src": e.get("src"), "dst": e.get("dst"), "rel": e.get("rel"), "created_by": e.get("created_by", []), "evidence_ids": e.get("evidence_ids", []), "confidence": e.get("confidence")}]))
    # dedupe by node sequence
    uniq = {}
    for p in paths:
        k = ">".join(n.get("key","") for n in p["nodes"])
        if k not in uniq or p["score"] > uniq[k]["score"]:
            uniq[k] = p
    ranked = sorted(uniq.values(), key=lambda x: x["score"], reverse=True)[:max_paths]
    return ranked
