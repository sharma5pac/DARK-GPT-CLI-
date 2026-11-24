"""Tool risk classification and permission policy.

Modes (config key ``permission_mode``):

* ``full``   - no prompts (default; matches previous DarkGPT behaviour)
* ``ask``    - confirm edits, shell commands and MCP calls
* ``strict`` - confirm every non-read-only action, including MCP

The TUI supplies the actual confirmation callback; headless callers can pass
``permission_callback=None`` and decisions fall back to the configured mode's
safe default (deny for write/shell in ask/strict, allow otherwise).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

READ_ONLY = {
    "read", "ls", "grep", "glob", "symbols", "find_refs",
    "git_status", "git_diff", "git_log", "git_blame",
    "lsp_definition", "lsp_references", "lsp_diagnostics",
    "web_search", "diff", "verify",
    # Cline-compatible read/analysis tools
    "read_files", "search_codebase", "fetch_web_content", "browser",
    "skills", "ask_question", "submit_and_exit",
    # Target graph/reporting (write to ~/.config/darkgpt, not workspace)
    "knowledge_graph", "evidence_log", "report_generate", "recon_orchestrate",
    "planner_update", "task_list", "task_wait", "task_log",
    "auth_vault", "proxy_control", "source_index", "blackboard", "attack_paths",
}
WRITE_TOOLS = {
    "write", "edit", "patch", "rename", "lsp_rename",
    "apply_patch", "editor",
}
WRITE_TOOLS = {
    "write", "edit", "patch", "rename", "lsp_rename",
    "apply_patch", "editor",
}

DANGEROUS = [
    r"\brm\s+-rf?\b", r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=",
    r"\bcurl\b[^\n]*\|\s*(ba)?sh", r"\bwget\b[^\n]*\|\s*(ba)?sh",
    r"\bchmod\s+777\b", r"\bgit\s+push\s+.*--force", r"\b:(?:\(\)\s*\{|\(){.*\|.*&.*};?\)\s*;?\s*\$",
    r">\s*/dev/sd[a-z]", r"\bshutdown\b", r"\breboot\b",
]
DANGEROUS_RE = re.compile("|".join(DANGEROUS), re.IGNORECASE)


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _command_strings(value: Any) -> List[str]:
    """Flatten Cline's run_commands union into plain strings for risk checks."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_command_strings(item))
        return out
    if isinstance(value, dict):
        if "commands" in value:
            return _command_strings(value["commands"])
        command = value.get("command", value.get("cmd"))
        return [str(command)] if command else []
    return []


def risk_of(tool: str, args: Dict[str, Any] | None = None) -> str:
    args = args or {}
    if tool == "shell":
        command = str(args.get("command") or "")
        return "danger" if DANGEROUS_RE.search(command) else "shell"
    if tool == "script":
        # A self-written script can do anything its interpreter can.
        code = str(args.get("code") or "")
        return "danger" if DANGEROUS_RE.search(code) else "shell"
    if tool == "run_commands":
        commands = " ".join(str(c) for c in _command_strings(args.get("commands")))
        return "danger" if DANGEROUS_RE.search(commands) else "shell"
    if tool == "task_start":
        kind = str(args.get("kind") or "shell").lower()
        if kind == "script":
            code = str(args.get("code") or "")
            return "danger" if DANGEROUS_RE.search(code) else "shell"
        command = str(args.get("command") or "")
        return "danger" if DANGEROUS_RE.search(command) else "shell"
    if tool == "discover_tools":
        return "safe"
    if tool == "autonomous_route":
        return "shell" if args.get("execute") else "safe"
    if tool == "task_kill":
        return "write"
    if tool in ("task_list", "task_wait", "task_log"):
        return "safe"
    if tool == "http_request":
        return "shell"  # network-active; can mutate remote state
    if tool == "oracle_validate":
        return "shell"  # makes network requests
    if tool == "verify_findings":
        return "shell"  # executes replay scripts
    if tool == "proxy_control":
        action = str(args.get("action") or "").lower()
        return "shell" if action in ("start", "stop", "replay") else "safe"
    if tool == "source_index":
        action = str(args.get("action") or "").lower()
        return "shell" if action == "fetch_js" else "safe"
    if tool in WRITE_TOOLS:
        # lsp_rename preview writes nothing.
        if tool == "lsp_rename" and not args.get("apply"):
            return "safe"
        return "write"
    if tool.startswith("mcp__"):
        return "mcp"
    if tool in ("delegate", "spawn_agent"):
        return "safe"  # workers are read-only and isolated
    if tool in READ_ONLY:
        return "safe"
    return "mcp"


def signature(tool: str, args: Dict[str, Any] | None = None) -> str:
    """Stable identifier for 'allow for this session' approvals."""
    args = args or {}
    key = tool
    if tool in WRITE_TOOLS or tool.startswith("mcp__"):
        key += "|" + str(args.get("path") or args.get("command") or "")
    elif tool == "shell":
        key += "|" + str(args.get("command") or "")
    elif tool == "script":
        digest = hashlib.sha1(str(args.get("code") or "").encode("utf-8", "replace")).hexdigest()[:12]
        key += "|" + digest
    elif tool == "autonomous_route":
        if args.get("execute"):
            commands = args.get("commands") or args.get("command") or ""
            digest = hashlib.sha1(str(commands).encode("utf-8", "replace")).hexdigest()[:12]
            key += "|" + digest
    elif tool == "apply_patch":
        digest = hashlib.sha1(str(args.get("input") or "").encode("utf-8", "replace")).hexdigest()[:12]
        key += "|" + digest
    elif tool == "run_commands":
        digest = hashlib.sha1("|".join(_command_strings(args.get("commands"))).encode("utf-8", "replace")).hexdigest()[:12]
        key += "|" + digest
    try:
        return json.dumps({"tool": key}, sort_keys=True)
    except Exception:
        return key


def needs_prompt(tool: str, args: Dict[str, Any] | None, mode: str, approved: set[str]) -> Tuple[bool, str, str]:
    risk = risk_of(tool, args)
    sig = signature(tool, args)
    if sig in approved:
        return False, risk, sig
    if risk == "safe":
        return False, risk, sig
    if mode == "full":
        return False, risk, sig
    if mode == "ask":
        return risk in ("write", "shell", "danger"), risk, sig
    if mode == "strict":
        return True, risk, sig
    return False, risk, sig


def default_allow(tool: str, args: Dict[str, Any] | None, mode: str) -> bool:
    """Fallback when no interactive callback exists."""
    risk = risk_of(tool, args)
    if mode == "full":
        return True
    if mode in ("ask", "strict"):
        return risk == "safe"
    return True


def describe(tool: str, args: Dict[str, Any] | None, risk: str) -> str:
    args = args or {}
    if tool == "shell":
        return f"Run shell command ({risk}):\n{str(args.get('command') or '')[:500]}"
    if tool == "script":
        return (
            f"Run self-written {args.get('language', 'python')} script ({risk}):\n"
            f"{str(args.get('code') or '')[:500]}"
        )
    if tool == "autonomous_route":
        objective = str(args.get("objective") or "")[:300]
        command = str(args.get("command") or "")
        if not command and args.get("commands"):
            command = str(args.get("commands"))[:300]
        return f"Autonomous route ({risk}): {objective}\n{command}"
    if tool in WRITE_TOOLS:
        return f"Modify file ({risk}):\n{args.get('path', '?')}"
    if tool == "run_commands":
        return "Run shell commands (" + risk + "):\n" + "\n".join(_command_strings(args.get("commands")))[:500]
    if tool == "task_start":
        kind = str(args.get("kind") or "shell").lower()
        if kind == "script":
            return f"Start background script ({risk}):\n{str(args.get('code') or '')[:500]}"
        return f"Start background command ({risk}):\n{str(args.get('command') or '')[:500]}"
    if tool == "task_kill":
        return f"Kill background task: {args.get('job_id')}"
    if tool == "fetch_web_content":
        urls = [str(r.get("url")) for r in _listify(args.get("requests")) if isinstance(r, dict)]
        return f"Fetch web content ({risk}):\n" + "\n".join(urls)[:500]
    if tool == "http_request":
        prof = f" auth_profile={args.get('auth_profile')}" if args.get('auth_profile') else ""
        return f"HTTP request ({risk}):\n{str(args.get('method') or 'GET').upper()} {str(args.get('url') or '')[:400]}{prof}"
    if tool == "oracle_validate":
        return f"Oracle validation ({risk}):\n{args.get('kind')} {args.get('url') or args.get('url_a') or ''}"
    if tool == "verify_findings":
        return f"Verify/replay findings ({risk}):\n{args.get('action')} {args.get('finding') or ''}"
    if tool == "proxy_control":
        return f"Proxy control ({risk}):\n{args.get('action')} target={args.get('target') or ''} port={args.get('port') or ''}"
    if tool == "source_index":
        return f"Source index ({risk}):\n{args.get('action')} {args.get('url') or args.get('paths') or ''}"
    if tool.startswith("mcp__"):
        return f"Call MCP tool ({risk}):\n{tool}"
    return f"Use {tool} ({risk})"
