"""Structured tool-output normalizers for DarkGPT.

Parses noisy command/tool output into compact deltas before they hit LLM context,
and optionally upserts entities into the target knowledge graph. Raw output is
still preserved by evidence/logging.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


_MAX_ITEMS = 40


def _lines(text: str) -> List[str]:
    return [l.rstrip("\n") for l in str(text or "").splitlines()]


def parse_nmap(text: str) -> Dict[str, Any]:
    hosts: List[Dict[str, Any]] = []
    services: List[Dict[str, Any]] = []
    current_host = ""
    for line in _lines(text):
        m = re.search(r"Nmap scan report for ([^\s]+)(?: \(([^)]+)\))?", line)
        if m:
            name = m.group(1)
            ip = m.group(2) or ""
            current_host = ip or name
            hosts.append({"host": current_host, "hostname": name if ip else ""})
            continue
        m = re.match(r"(\d+)\/(tcp|udp)\s+open\s+(\S+)(?:\s+(.*))?", line.strip())
        if m and current_host:
            services.append({
                "host": current_host,
                "port": int(m.group(1)),
                "proto": m.group(2),
                "name": m.group(3),
                "detail": (m.group(4) or "").strip()[:200],
            })
    return {"kind": "nmap", "hosts": _dedupe(hosts, ["host"]), "services": _dedupe(services, ["host", "port", "proto"])}


def parse_ffuf(text: str) -> Dict[str, Any]:
    endpoints = []
    # json lines or ffuf json
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            for r in obj["results"][:_MAX_ITEMS]:
                if not isinstance(r, dict):
                    continue
                url = str(r.get("url") or "")
                status = r.get("status")
                length = r.get("length")
                words = r.get("words")
                if url:
                    endpoints.append({"url": url, "status": status, "length": length, "words": words})
            return {"kind": "ffuf", "endpoints": endpoints}
    except Exception:
        pass
    for line in _lines(text):
        m = re.search(r"(https?://\S+)\s+\[Status:\s*(\d+),\s*Size:\s*(\d+),\s*Words:\s*(\d+)", line)
        if m:
            endpoints.append({"url": m.group(1), "status": int(m.group(2)), "length": int(m.group(3)), "words": int(m.group(4))})
        else:
            m = re.search(r"(\S+)\s+\[Status:\s*(\d+)", line)
            if m and m.group(1).startswith("/"):
                endpoints.append({"path": m.group(1), "status": int(m.group(2))})
    return {"kind": "ffuf", "endpoints": _dedupe(endpoints, ["url", "path", "status"])[:_MAX_ITEMS]}


def parse_httpx(text: str) -> Dict[str, Any]:
    endpoints = []
    try:
        # jsonl
        for line in _lines(text):
            line = line.strip()
            if not line.startswith("{"):
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                endpoints.append({
                    "url": obj.get("url") or obj.get("input"),
                    "status": obj.get("status_code") or obj.get("status"),
                    "title": str(obj.get("title") or "")[:120],
                    "tech": obj.get("tech") or obj.get("technologies") or [],
                    "webserver": obj.get("webserver"),
                })
        if endpoints:
            return {"kind": "httpx", "endpoints": endpoints[:_MAX_ITEMS]}
    except Exception:
        pass
    for line in _lines(text):
        m = re.search(r"(https?://\S+)\s+\[(\d+)\]", line)
        if m:
            endpoints.append({"url": m.group(1), "status": int(m.group(2))})
    return {"kind": "httpx", "endpoints": _dedupe(endpoints, ["url"])[:_MAX_ITEMS]}


def parse_nuclei(text: str) -> Dict[str, Any]:
    findings = []
    for line in _lines(text):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and (obj.get("template-id") or obj.get("info")):
                info = obj.get("info") or {}
                findings.append({
                    "title": str(info.get("name") or obj.get("template-id") or "nuclei finding"),
                    "severity": str(info.get("severity") or "info").lower(),
                    "asset": obj.get("matched-at") or obj.get("host") or "",
                    "details": str(obj.get("matcher-name") or info.get("description") or "")[:500],
                })
                continue
        except Exception:
            pass
        m = re.match(r"\[(?P<tpl>[^\]]+)\]\s+\[(?P<proto>[^\]]+)\]\s+\[(?P<sev>[^\]]+)\]\s+(?P<target>\S+)", line)
        if m:
            findings.append({
                "title": m.group("tpl"),
                "severity": m.group("sev").lower(),
                "asset": m.group("target"),
                "details": m.group("proto"),
            })
    return {"kind": "nuclei", "findings": findings[:_MAX_ITEMS]}


def _dedupe(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        sig = "|".join(str(it.get(k, "")).strip().lower() for k in keys)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        out.append(it)
    return out


def detect_and_parse(name: str, args: Dict[str, Any], result: str) -> Optional[Dict[str, Any]]:
    text = str(result or "")
    cmd = str((args or {}).get("command") or "")
    hay = f"{name} {cmd} {text[:2000]}".lower()
    if "nmap" in hay and ("open" in text or "nmap scan report" in text.lower()):
        parsed = parse_nmap(text)
        if parsed["hosts"] or parsed["services"]:
            return parsed
    if "ffuf" in hay and ("[status:" in text.lower() or '"results"' in text):
        parsed = parse_ffuf(text)
        if parsed["endpoints"]:
            return parsed
    if "httpx" in hay and ("http" in text):
        parsed = parse_httpx(text)
        if parsed["endpoints"]:
            return parsed
    if "nuclei" in hay and ("[info]" in text.lower() or "[high]" in text.lower() or "[critical]" in text.lower() or "[medium]" in text.lower() or '"template-id"' in text):
        parsed = parse_nuclei(text)
        if parsed["findings"]:
            return parsed
    return None


def compact_for_context(name: str, args: Dict[str, Any], result: str, limit: int = 6000) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return (possibly compacted result for LLM, parsed delta or None)."""
    parsed = detect_and_parse(name, args, result)
    if not parsed:
        return result, None
    summary = {"parsed": parsed, "raw_preview": str(result)[:1200], "note": "Raw output preserved in evidence/log; parsed delta auto-upserted to KG when target active."}
    compacted = json.dumps(summary, ensure_ascii=False)
    if len(compacted) > limit:
        compacted = compacted[:limit] + "...[truncated]"
    return compacted, parsed


def upsert_to_graph(target: str, parsed: Dict[str, Any], source: str = "tool_parser") -> Dict[str, int]:
    from agent import knowledge_graph as kg
    counts = {"hosts": 0, "services": 0, "endpoints": 0, "findings": 0}
    kind = parsed.get("kind")
    for h in parsed.get("hosts") or []:
        host = str(h.get("host") or "").strip()
        if host:
            kg.add_host(target, host, {k: v for k, v in h.items() if k != "host"}, source=source)
            counts["hosts"] += 1
    for s in parsed.get("services") or []:
        host = str(s.get("host") or "").strip(); port = s.get("port")
        if host and port:
            kg.add_service(target, host, int(port), str(s.get("proto") or "tcp"), {k: v for k, v in s.items() if k not in {"host","port","proto"}}, source=source)
            counts["services"] += 1
    for e in parsed.get("endpoints") or []:
        url = str(e.get("url") or "").strip()
        if url:
            kg.add_endpoint(target, url, {k: v for k, v in e.items() if k != "url"}, source=source)
            counts["endpoints"] += 1
        else:
            path = str(e.get("path") or "").strip()
            if path:
                counts["endpoints"] += 0
    for f in parsed.get("findings") or []:
        title = str(f.get("title") or "").strip()
        if title:
            kg.add_finding(target, title, str(f.get("severity") or "info"), str(f.get("details") or ""), "", {"asset": f.get("asset", ""), "status": "attempted"}, source=source)
            counts["findings"] += 1
    return counts
