"""Auto-discover local pentest tools and map them to a user objective.

The agent should never assume only the tools it already knows about. This module
builds a live inventory from PATH and ~/.local/bin, then scores each binary
against the user's task so an objective like "password and encryption auditing
with aircrack-ng" automatically produces aircrack-ng, hashcat, john, and the
matching Kala/GUI routes.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# capability catalog - deliberately broad; discovery still verifies existence.
TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "aircrack-ng": {"category": "password & encryption auditing", "tags": ["password", "encryption", "wireless", "wpa", "wep", "crack", "capture"], "gui": False, "aliases": ["aircrack"]},
    "airmon-ng": {"category": "wireless auditing", "tags": ["wireless", "monitor", "capture", "wifi"], "gui": False, "aliases": []},
    "airodump-ng": {"category": "wireless auditing", "tags": ["wireless", "capture", "wifi", "recon"], "gui": False, "aliases": []},
    "aireplay-ng": {"category": "wireless auditing", "tags": ["wireless", "wifi", "inject", "deauth"], "gui": False, "aliases": []},
    "wifite": {"category": "wireless auditing", "tags": ["wireless", "wifi", "password", "audit"], "gui": False, "aliases": []},
    "hashcat": {"category": "password & encryption auditing", "tags": ["password", "hash", "encryption", "crack", "gpu", "bruteforce"], "gui": False, "aliases": []},
    "john": {"category": "password & encryption auditing", "tags": ["password", "hash", "encryption", "crack", "cpu", "bruteforce"], "gui": False, "aliases": ["johntheripper", "john the ripper"]},
    "hash-identifier": {"category": "password & encryption auditing", "tags": ["password", "hash", "encryption", "identify", "crypto"], "gui": False, "aliases": ["hashid"]},
    "openssl": {"category": "encryption auditing", "tags": ["encryption", "crypto", "tls", "certificate", "cipher"], "gui": False, "aliases": []},
    "hydra": {"category": "password auditing", "tags": ["password", "login", "bruteforce", "auth", "crack"], "gui": False, "aliases": ["thc-hydra"]},
    "medusa": {"category": "password auditing", "tags": ["password", "login", "bruteforce", "auth", "crack"], "gui": False, "aliases": []},
    "crackmapexec": {"category": "active directory auditing", "tags": ["password", "auth", "active directory", "smb", "crack"], "gui": False, "aliases": ["cme"]},
    "nmap": {"category": "network recon", "tags": ["recon", "ports", "services", "network", "scan"], "gui": False, "aliases": []},
    "zenmap": {"category": "network recon", "tags": ["recon", "ports", "services", "network", "gui", "scan"], "gui": True, "aliases": []},
    "masscan": {"category": "network recon", "tags": ["recon", "ports", "network", "fast", "scan"], "gui": False, "aliases": []},
    "subfinder": {"category": "recon", "tags": ["recon", "subdomains", "dns", "osint"], "gui": False, "aliases": []},
    "amass": {"category": "recon", "tags": ["recon", "subdomains", "dns", "osint"], "gui": False, "aliases": []},
    "theharvester": {"category": "recon", "tags": ["recon", "emails", "subdomains", "osint"], "gui": False, "aliases": ["theHarvester"]},
    "ffuf": {"category": "web auditing", "tags": ["web", "fuzz", "content", "directory", "param"], "gui": False, "aliases": []},
    "gobuster": {"category": "web auditing", "tags": ["web", "fuzz", "content", "directory", "dns"], "gui": False, "aliases": []},
    "dirb": {"category": "web auditing", "tags": ["web", "content", "directory", "bruteforce"], "gui": False, "aliases": []},
    "wfuzz": {"category": "web auditing", "tags": ["web", "fuzz", "param", "injection"], "gui": False, "aliases": []},
    "sqlmap": {"category": "database auditing", "tags": ["web", "sqli", "database", "injection", "sql"], "gui": False, "aliases": []},
    "commix": {"category": "web auditing", "tags": ["web", "command injection", "rce", "injection"], "gui": False, "aliases": []},
    "nikto": {"category": "web auditing", "tags": ["web", "scan", "misconfig"], "gui": False, "aliases": []},
    "curl": {"category": "http client", "tags": ["http", "web", "request", "fetch", "api"], "gui": False, "aliases": []},
    "wget": {"category": "http client", "tags": ["http", "web", "download", "fetch"], "gui": False, "aliases": []},
    "burpsuite": {"category": "web proxy / gui", "tags": ["web", "proxy", "intercept", "repeater", "intruder", "gui"], "gui": True, "aliases": ["burpsuite-atk", "burp"]},
    "wireshark": {"category": "packet analysis", "tags": ["network", "capture", "packets", "protocol", "gui"], "gui": True, "aliases": ["tshark"]},
    "tcpdump": {"category": "packet analysis", "tags": ["network", "capture", "packets", "cli"], "gui": False, "aliases": []},
    "jsql-injection": {"category": "database auditing", "tags": ["web", "sqli", "database", "gui", "injection"], "gui": True, "aliases": ["jsql"]},
    "recon-ng": {"category": "recon", "tags": ["recon", "osint", "web", "gui"], "gui": True, "aliases": []},
    "ghidra": {"category": "reverse engineering", "tags": ["reverse", "binary", "decompile", "gui", "malware"], "gui": True, "aliases": []},
    "binwalk": {"category": "reverse engineering", "tags": ["reverse", "firmware", "binary", "extract"], "gui": False, "aliases": []},
    "strings": {"category": "reverse engineering", "tags": ["reverse", "binary", "strings", "static"], "gui": False, "aliases": []},
    "volatility": {"category": "forensics", "tags": ["forensics", "memory", "ram", "incident"], "gui": False, "aliases": ["volatility3"]},
    "bloodhound": {"category": "active directory auditing", "tags": ["active directory", "graph", "gui", "ad"], "gui": True, "aliases": []},
    "bettercap": {"category": "network interception", "tags": ["network", "mitm", "capture", "wifi"], "gui": False, "aliases": []},
    "responder": {"category": "network interception", "tags": ["network", "auth", "llmnr", "poison"], "gui": False, "aliases": []},
}

_WORD_RE = re.compile(r"[a-z0-9+_.-]+")


def _norm(text: Any) -> str:
    return str(text or "").lower().strip()


def _catalog_entry(name: str) -> Dict[str, Any]:
    canonical = _norm(name)
    for key, meta in TOOL_CATALOG.items():
        if canonical == key or canonical in [_norm(a) for a in meta.get("aliases", [])]:
            entry = dict(meta)
            entry["name"] = key
            return entry
    return {
        "name": name,
        "category": "general",
        "tags": [],
        "gui": False,
        "aliases": [],
    }


def discover_tools(extra_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Return verified local tools with capability tags and resolved paths."""
    names = set(TOOL_CATALOG)
    for item in extra_names or []:
        if str(item).strip():
            names.add(str(item).strip())
    found: Dict[str, Dict[str, Any]] = {}
    local_bin = Path.home() / ".local" / "bin"
    for name in sorted(names):
        path = shutil.which(name)
        if not path and local_bin.is_dir():
            candidate = local_bin / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                path = str(candidate)
        if not path:
            continue
        entry = _catalog_entry(name)
        entry.update({"path": path, "kind": "binary", "available": True})
        found.setdefault(entry["name"], entry)
    return sorted(found.values(), key=lambda x: x["name"].lower())


def mcp_names_for_tools(tools: Sequence[Dict[str, Any]], mcp_tool_names: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Map local tool names to matching Kala/MCP tool names when present."""
    mapping: Dict[str, str] = {}
    for tool in tools:
        needle = re.sub(r"[^a-z0-9]", "", _norm(tool["name"]))
        for mcp_name in mcp_tool_names or []:
            compact = re.sub(r"[^a-z0-9]", "", _norm(mcp_name))
            if needle and needle in compact:
                mapping[tool["name"]] = str(mcp_name)
                break
    return mapping


def rank_tools(
    task: str,
    tools: Sequence[Dict[str, Any]],
    *,
    mcp_tool_names: Optional[Iterable[str]] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Score discovered tools against the objective and explain each match."""
    text = _norm(task)
    words = set(_WORD_RE.findall(text))
    mcp_map = mcp_names_for_tools(tools, mcp_tool_names)
    ranked: List[Dict[str, Any]] = []
    for tool in tools:
        score = 0
        reasons: List[str] = []
        names = [tool["name"], *tool.get("aliases", [])]
        for name in names:
            needle = _norm(name)
            if needle and needle in text:
                score += 120
                reasons.append(f"explicit tool name: {name}")
        for tag in tool.get("tags", []):
            tag_norm = _norm(tag)
            if tag_norm and tag_norm in text:
                score += 14
                reasons.append(f"tag {tag}")
            elif tag_norm and tag_norm in words:
                score += 10
        category = _norm(tool.get("category"))
        if category and category in text:
            score += 18
            reasons.append(f"category {tool['category']}")
        else:
            for word in _WORD_RE.findall(category):
                if word in words:
                    score += 6
                    reasons.append(f"category word {word}")
        mcp = mcp_map.get(tool["name"])
        if mcp:
            score += 35
            reasons.append(f"Kala/MCP wrapper: {mcp}")
        if "gui" in words and tool.get("gui"):
            score += 20
            reasons.append("GUI route available")
        if score <= 0:
            continue
        ranked.append({
            **tool,
            "score": score,
            "reason": "; ".join(dict.fromkeys(reasons))[:300],
            "mcp_tool": mcp,
        })
    ranked.sort(key=lambda x: (-x["score"], x["name"].lower()))
    return ranked[: max(0, limit)]


def build_routes(task: str, ranked: Sequence[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """Produce bounded, ordered strategies for the top matching tools."""
    routes: List[Dict[str, Any]] = []
    for tool in ranked[: max(0, limit)]:
        name = tool["name"]
        if tool.get("mcp_tool"):
            routes.append({
                "route": "kala",
                "tool": name,
                "mcp_tool": tool["mcp_tool"],
                "priority": 1,
                "next": f"call {tool['mcp_tool']} with task-derived options",
            })
        routes.append({
            "route": "cli",
            "tool": name,
            "priority": 2,
            "next": f"run {name} --help, then build a minimal command for: {task[:120]}",
            "probe": f"{name} --help",
        })
        if tool.get("gui"):
            app = "burpsuite-atk" if name == "burpsuite" else name
            routes.append({
                "route": "gui",
                "tool": name,
                "priority": 3,
                "next": f"desktop_launch({app}) -> desktop_wait -> desktop_snapshot -> click/type -> verify",
            })
        routes.append({
            "route": "script",
            "tool": name,
            "priority": 4,
            "next": "write a bounded helper script that parses output and records evidence",
        })
    routes.sort(key=lambda x: x["priority"])
    return routes


def inventory_summary(tools: Sequence[Dict[str, Any]]) -> str:
    if not tools:
        return "no matching local tools"
    return ", ".join(f"{t['name']} ({t['category']})" for t in tools[:12])
