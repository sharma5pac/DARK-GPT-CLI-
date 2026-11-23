"""White-box/source-to-sink correlation for DarkGPT.

Indexes local source files, fetched JS bundles, and extracted endpoint/param/sink
hints into the target knowledge graph. Lightweight static heuristics - not a full
taint engine - but enough to turn blind fuzzing into targeted validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from agent import knowledge_graph as kg

JS_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte"}
SOURCE_EXT = JS_EXT | {".py", ".php", ".rb", ".java", ".go", ".cs", ".json", ".yaml", ".yml"}
ENDPOINT_RE = re.compile(r"""(?:"|')(?:/(?:api|v\d+|graphql|admin|internal|auth|login|logout|upload|user|account|token|oauth|callback)[^"'`\s]{0,200}|https?://[^"'`\s]{4,300})(?:"|')""", re.I)
FETCH_RE = re.compile(r"""(?:fetch|axios(?:\.[a-z]+)?|XMLHttpRequest|\.open\(|\.get\(|\.post\(|\.put\(|\.patch\(|\.delete\(|request\()\s*\(?\s*["'`]([^"'`\s]{1,300})""", re.I)
PARAM_RE = re.compile(r"""(?:[?&]|(?:name|param|key|field|query)\s*[:=]\s*["'])\s*([a-zA-Z_][a-zA-Z0-9_\-]{1,64})""")
SINK_PATTERNS = {
    "dom_xss": re.compile(r"\b(innerHTML|outerHTML|document\.write|insertAdjacentHTML|eval\(|setTimeout\(\s*[\"'`]|setInterval\(\s*[\"'`])"),
    "redirect": re.compile(r"\b(location(?:\.href|\.assign|\.replace)?\s*=|window\.open\()"),
    "sql_like": re.compile(r"\b(SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|execute\(|cursor\.execute)\b", re.I),
    "cmd_exec": re.compile(r"\b(os\.system|subprocess\.|child_process|exec\(|spawn\(|Runtime\.getRuntime)", re.I),
    "ssrf_fetch": re.compile(r"\b(requests\.(get|post)|urllib\.request|http\.get|fetch\(|axios\.)", re.I),
    "path_file": re.compile(r"\b(readFile|writeFile|open\(|Path\(|send_file|res\.sendFile)", re.I),
}


def _read_text(path: Path, max_bytes: int = 300000) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_endpoints(text: str, base_url: str = "") -> List[Dict[str, Any]]:
    found = []
    for m in ENDPOINT_RE.finditer(text):
        raw = m.group(0).strip("'\"")
        url = raw
        if base_url and raw.startswith("/"):
            url = urljoin(base_url.rstrip("/") + "/", raw.lstrip("/"))
        found.append({"url": url, "raw": raw})
    for m in FETCH_RE.finditer(text):
        raw = m.group(1).strip()
        if not raw or raw.startswith(("data:", "javascript:", "#")):
            continue
        url = urljoin(base_url.rstrip("/") + "/", raw) if base_url and raw.startswith("/") else raw
        found.append({"url": url, "raw": raw})
    # dedupe
    seen = set(); out = []
    for e in found:
        k = e["url"].split("#")[0]
        if k and k not in seen:
            seen.add(k); out.append(e)
    return out[:200]


def _extract_params(text: str) -> List[str]:
    seen = set(); out = []
    for m in PARAM_RE.finditer(text):
        p = m.group(1)
        k = p.lower()
        if k not in seen and len(p) <= 64:
            seen.add(k); out.append(p)
    return out[:100]


def _extract_sinks(text: str) -> List[Dict[str, Any]]:
    hits = []
    for kind, pat in SINK_PATTERNS.items():
        for m in pat.finditer(text):
            line = text[:m.start()].count("\n") + 1
            snippet = text[max(0, m.start()-40):m.end()+80].replace("\n", " ")[:160]
            hits.append({"kind": kind, "line": line, "snippet": snippet})
            if len(hits) >= 50:
                return hits
    return hits


def index_file(target: str, path: str, base_url: str = "", source: str = "source_correlation") -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": f"not found: {p}"}
    text = _read_text(p)
    if not text.strip():
        return {"ok": False, "error": f"empty/unreadable: {p}"}
    endpoints = _extract_endpoints(text, base_url=base_url)
    params = _extract_params(text)
    sinks = _extract_sinks(text)
    rel = str(p)
    try:
        rel = str(p.relative_to(Path.cwd()))
    except Exception:
        pass
    artifact = kg.upsert_node(target, "source_artifact", rel, {
        "path": rel, "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
        "bytes": len(text.encode("utf-8", "replace")), "kind": p.suffix.lower(),
    }, source=source)
    counts = {"endpoints": 0, "params": 0, "sinks": 0}
    for ep in endpoints:
        url = ep["url"]
        if url.startswith(("http://", "https://", "/")):
            node = kg.add_endpoint(target, url if url.startswith("http") else urljoin(base_url.rstrip("/")+"/", url.lstrip("/")) if base_url else url, {"discovered_from": rel}, source=source)
            kg.add_edge(target, artifact["key"], node["key"], "references")
            counts["endpoints"] += 1
    for param in params[:50]:
        node = kg.upsert_node(target, "param", f"{rel}:{param}", {"name": param, "location": "unknown", "source_file": rel}, source=source)
        kg.add_edge(target, artifact["key"], node["key"], "has_param")
        counts["params"] += 1
    for sink in sinks[:30]:
        ident = f"{rel}:{sink['kind']}:{sink['line']}"
        node = kg.upsert_node(target, "sink", ident, {"kind": sink["kind"], "line": sink["line"], "snippet": sink["snippet"], "source_file": rel}, source=source)
        kg.add_edge(target, artifact["key"], node["key"], "has_sink")
        counts["sinks"] += 1
    return {"ok": True, "path": rel, "counts": counts, "sample_endpoints": [e["url"] for e in endpoints[:10]], "sample_sinks": sinks[:5]}


def index_paths(target: str, paths: List[str], base_url: str = "", limit: int = 40) -> Dict[str, Any]:
    results = []
    totals = {"files": 0, "endpoints": 0, "params": 0, "sinks": 0, "errors": 0}
    for raw in (paths or [])[:max(1, limit)]:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if len(results) >= limit: break
                if child.is_file() and child.suffix.lower() in SOURCE_EXT and ".git" not in child.parts and "node_modules" not in child.parts:
                    r = index_file(target, str(child), base_url=base_url)
                    results.append(r)
        else:
            r = index_file(target, str(p), base_url=base_url)
            results.append(r)
    for r in results:
        if r.get("ok"):
            totals["files"] += 1
            for k in ("endpoints", "params", "sinks"):
                totals[k] += int((r.get("counts") or {}).get(k, 0))
        else:
            totals["errors"] += 1
    return {"ok": True, "target": target, "base_url": base_url, "totals": totals, "results": results[:20]}


def fetch_and_index_js(target: str, url: str, timeout: int = 20) -> Dict[str, Any]:
    import requests
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "DarkGPT/1.0"}, allow_redirects=True)
        text = r.text or ""
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}
    cache_dir = kg.target_dir(target) / "source_artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", urlparse(url).path or "index.js").strip("_")[-120] or "bundle.js"
    if not re.search(r"\.[a-z0-9]{1,5}$", name, re.I):
        name += ".js"
    p = cache_dir / name
    p.write_text(text, encoding="utf-8", errors="replace")
    indexed = index_file(target, str(p), base_url=url, source="js_bundle")
    indexed["url"] = url
    indexed["saved_to"] = str(p)
    indexed["status_code"] = getattr(r, "status_code", None)
    return indexed
