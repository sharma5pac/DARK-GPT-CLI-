"""Streaming agent loop with monotonic phase telemetry and unified tool routing.

Every duration reported to the UI is measured with ``time.perf_counter()`` at
the exact lifecycle boundary:

* connecting  - request dispatch -> first SSE event
* thinking    - first reasoning token -> first content/tool token (or stream end)
* writing     - first content token -> stream end
* tool        - tool dispatch -> tool result

The UI never has to guess or back-calculate these values.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

from utils.get_response import (
    OpenRouterError,
    auth_headers,
    chat_url,
    parse_error_response,
    post_with_retry,
    provider_config,
)
from utils.search import SearchError, format_results, web_search
from utils import appconfig, hooks, permissions, scripting, undo, vcs, verify
from agent import (
    browser,
    cline_tools,
    codeintel,
    context as context_util,
    evidence as evidence_util,
    knowledge_graph as kg_util,
    planner as planner_util,
    lsp,
    memory as memory_util,
    reporting as reporting_util,
    review as review_util,
    routing,
    skills as skills_util,
    subagents,
    tool_parsers,
    auth_vault,
    source_correlation,
    proxy as proxy_util,
)
from agent.jobs import JobManager

MAX_TOOL_OUTPUT = 12000
DEFAULT_MAX_STEPS = int(appconfig.get("max_steps", 28) or 28)
MAX_REPEAT_CALLS = 3
DEFAULT_TOOLS_PER_TURN = int(appconfig.get("tools_per_turn", 40) or 40)
MAX_VERIFY_ROUNDS = 2
MAX_REVIEW_ROUNDS = 1
MAX_SUBAGENTS_PER_TURN = 3
MAX_NATIVE_RECOVERIES = 2
NATIVE_RECOVERY_TOOLS = ("shell", "read", "write", "edit", "grep", "glob", "ls")
EDIT_TOOLS = ("write", "edit", "patch", "rename", "lsp_rename", "apply_patch", "editor")
CLINE_TOOLS = (
    "read_files", "search_codebase", "run_commands", "fetch_web_content",
    "apply_patch", "editor", "skills", "ask_question", "submit_and_exit", "spawn_agent",
)
BROWSER_TOOLS = ("browser",)
DEFAULT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition"


def _dur(start: float, end: float) -> float:
    """Monotonic elapsed time, never negative even on synthetic clocks."""
    return max(0.0, end - start)


def clamp(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text)-limit} chars]"


def safe_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return p.resolve()


def clean_network_error(exc: BaseException) -> str:
    """Describe a transport failure using the *active* provider, not OpenRouter.

    The same agent loop serves both OpenRouter and NVIDIA NIM. Hard-coding
    OpenRouter here produced misleading errors after switching backends.
    """
    text = str(exc) or exc.__class__.__name__
    lowered = text.lower()
    cfg = provider_config()
    label = cfg["label"]
    host = urlparse(chat_url()).netloc or label
    dns_failure = any(marker in lowered for marker in (
        "nameresolution", "name resolution", "getaddrinfo", "failed to resolve",
        "name or service not known", "temporary failure in name resolution",
        "nodename nor servname", "no address associated with hostname",
    ))
    if dns_failure:
        return f"DNS lookup failed for {host}. Check your network or VPN, then retry."
    if "connection refused" in lowered:
        return f"Connection refused by {host}. Check your network/proxy, then retry."
    # Check timeouts before a generic "dns lookup" mention: a slow resolver
    # must be reported as a timeout, not as a permanent resolution failure.
    if "timed out" in lowered or "timeout" in lowered:
        return f"The connection to {host} timed out. Check your network, then retry."
    if "dns lookup" in lowered:
        return f"DNS lookup failed for {host}. Check your network or VPN, then retry."
    return f"Network error reaching {label} ({host}): {text[:240]}"


_TRANSIENT_MARKERS = (
    "timed out", "timeout", "temporarily unavailable", "connection reset",
    "connection refused", "network", "dns", "econnreset", "econnrefused",
    "rate limit", "429", "502", "503", "504", "try again",
)


def _short_error(result: str) -> str:
    """First meaningful line of a failed tool result, JSON-aware."""
    text = str(result or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("error", "detail", "message", "hint"):
                value = data.get(key)
                if value:
                    text = str(value)
                    break
    except Exception:
        pass
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:240]


def _is_transient(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _error_signature(text: str) -> str:
    """Coarse signature so the same failure class groups together.

    Digits, addresses and single-letter path fragments are normalised away so
    'HTTP 500 at /a' and 'HTTP 502 at /b' share one signature.
    """
    lowered = (text or "").lower()
    lowered = re.sub(r"0x[0-9a-f]+", "0xX", lowered)
    lowered = re.sub(r"\d+", "N", lowered)
    return " ".join(re.findall(r"[a-z]{2,}", lowered))[:120]


def edit_applied(name: str, result: str) -> bool:
    """True when a tool actually changed files (lsp_rename can be preview-only)."""
    if name not in EDIT_TOOLS:
        return False
    if name != "lsp_rename":
        return True
    try:
        data = json.loads(result)
        return bool(isinstance(data, dict) and data.get("apply"))
    except Exception:
        return False


def should_verify(name: str, result: str, tool_ok: bool) -> bool:
    """A self-written script can change files, so it gets the same verify gate."""
    if not tool_ok:
        return False
    return edit_applied(name, result) or name == "script"


def is_background_tool(name: str) -> bool:
    """Tools whose raw output should stay out of the chat stream by default."""
    return name.startswith("mcp__") or name in ("web_search",)


def is_tool_support_error(message: str) -> bool:
    """True only for a genuine 'this route cannot do tool calling' error.

    A bare ``no endpoints found`` is deliberately NOT enough: that is also
    returned for stale model ids and transient routing gaps, and used to drop
    perfectly capable models into ReAct mode on a single unlucky request.
    """
    lowered = (message or "").lower()
    return any(phrase in lowered for phrase in (
        "does not support tool",
        "does not support function",
        "tool use is not supported",
        "tool calling is not supported",
        "support tool use",
        "support function calling",
        "tool_calls is not supported",
        "tools are not supported",
        "no endpoints found that support tool",
        "no endpoint supports tool",
    ))


def validate_tool_args(name: str, args: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Optional[str]:
    """Lightweight JSON-schema check: required keys and primitive types.

    Returns a human-readable problem, or None when the arguments are usable.
    This runs before execution so a malformed model call never burns a tool
    slot or produces a confusing downstream error.
    """
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        return None
    if not isinstance(args, dict):
        return f"{name}: arguments must be an object, got {type(args).__name__}"
    for key in schema.get("required") or []:
        if key not in args or args[key] in (None, ""):
            return f"{name}: missing required argument '{key}'"
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return None
    for key, value in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        want = spec.get("type")
        ok = True
        if want == "string":
            ok = isinstance(value, str)
        elif want == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif want == "number":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif want == "boolean":
            ok = isinstance(value, bool)
        elif want == "array":
            ok = isinstance(value, list)
        elif want == "object":
            ok = isinstance(value, dict)
        if not ok:
            return f"{name}: argument '{key}' should be {want}, got {type(value).__name__}"
    return None


def _pre_event(risk: str) -> str:
    if risk == "write":
        return "pre_edit"
    if risk in ("shell", "danger"):
        return "pre_shell"
    return "pre_tool"


def _post_event(risk: str) -> str:
    if risk == "write":
        return "post_edit"
    if risk in ("shell", "danger"):
        return "post_shell"
    return "post_tool"


class ToolRegistry:
    def __init__(
        self,
        mcp_manager=None,
        disabled: Optional[set[str]] = None,
        disabled_servers: Optional[set[str]] = None,
        subagent_pool=None,
        permission_callback=None,
        permission_mode: str = "full",
        hooks_enabled: bool = True,
        skills=None,
        ask_callback=None,
        model: str = "",
    ):
        self.mcp = mcp_manager
        self.disabled = disabled or set()
        self.disabled_servers = disabled_servers or set()
        self.subagents = subagent_pool
        self.model = model
        self.permission_callback = permission_callback
        self.permission_mode = permission_mode
        self.hooks_enabled = hooks_enabled
        self.skills = skills
        self.ask_callback = ask_callback
        self.approved: set[str] = set()
        self.jobs = JobManager()
        self.proxy = proxy_util.ProxyManager()
        self._telemetry: Dict[str, Any] = {"tool_calls": 0, "llm_calls": 0, "input_chars": 0, "output_chars": 0}

    def local_specs(self) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "shell", "description": "Run a bash command in cwd", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read", "description": "Read a file with line numbers", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "edit", "description": "Edit a file: replace the first occurrence of old with new. old may also be a line range like '55-64'.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
            {"type": "function", "function": {"name": "ls", "description": "List directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
            {"type": "function", "function": {"name": "grep", "description": "Search file contents", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "glob", "description": "Find files by glob", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "web_search", "description": "Search web for current info/sources", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "symbols", "description": "List functions/classes/symbols in a source file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "find_refs", "description": "Find references to a symbol in the workspace", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "path": {"type": "string"}}, "required": ["symbol"]}}},
            {"type": "function", "function": {"name": "rename", "description": "Rename a symbol across code files (with diff preview via dry_run)", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "new_name": {"type": "string"}, "path": {"type": "string"}, "dry_run": {"type": "boolean"}}, "required": ["symbol", "new_name"]}}},
            {"type": "function", "function": {"name": "patch", "description": "Apply several exact-text replacements atomically", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "hunks": {"type": "array"}}, "required": ["path", "hunks"]}}},
            {"type": "function", "function": {"name": "git_status", "description": "Show git working tree status", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "git_diff", "description": "Show the uncommitted git diff", "parameters": {"type": "object", "properties": {"staged": {"type": "boolean"}}, "required": []}}},
            {"type": "function", "function": {"name": "git_log", "description": "Show recent git commits", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}}},
            {"type": "function", "function": {"name": "git_blame", "description": "Show blame for a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "lsp_definition", "description": "Go to a symbol's definition using a real language server (pyright/tsserver/rust-analyzer/gopls); falls back to regex search", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "symbol": {"type": "string"}, "line": {"type": "integer"}, "character": {"type": "integer"}}, "required": ["path", "symbol"]}}},
            {"type": "function", "function": {"name": "lsp_references", "description": "Find all references to a symbol via LSP (semantic, not textual)", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "symbol": {"type": "string"}, "line": {"type": "integer"}, "character": {"type": "integer"}}, "required": ["path", "symbol"]}}},
            {"type": "function", "function": {"name": "lsp_diagnostics", "description": "Get type/compile diagnostics (errors, warnings) for a file from the language server or project compiler", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "lsp_rename", "description": "Semantic rename across the workspace using LSP. Preview by default; pass apply=true to write changes", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "symbol": {"type": "string"}, "new_name": {"type": "string"}, "line": {"type": "integer"}, "character": {"type": "integer"}, "apply": {"type": "boolean"}}, "required": ["path", "symbol", "new_name"]}}},
            {"type": "function", "function": {"name": "delegate", "description": "Spawn an isolated read-only subagent (explore/review/test) to work on a focused task in parallel; returns a concise report", "parameters": {"type": "object", "properties": {"role": {"type": "string", "enum": ["explore", "review", "test", "recon", "vuln_analysis", "exploit_validation", "reporting"]}, "task": {"type": "string"}, "focus": {"type": "string"}}, "required": ["role", "task"]}}},
            {"type": "function", "function": {"name": "script", "description": "PREFERRED for speed: write and RUN a Python or bash helper script in one call. Use this first for recon/enumeration/fuzzing/parsing/bulk checks/multi-step automation/report building/data transforms-anything that would take multiple shell/read/grep/browser calls. Scripts are faster, repeatable, and saved as evidence under ~/.cache/darkgpt/scripts. Return exit_code, output, script_path. Use python for parsing/HTTP/JSON/logic; bash for quick CLI pipelines. Keep scripts focused; print clear findings.", "parameters": {"type": "object", "properties": {"language": {"type": "string", "enum": ["python", "bash"]}, "code": {"type": "string", "description": "Complete runnable script. Include imports/shebang assumptions as needed; print actionable output."}, "args": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "integer", "description": "Seconds, default 30, max 120."}}, "required": ["language", "code"]}}},
            # ---- Cline-compatible built-in tools ---------------------------------
            {"type": "function", "function": {"name": "read_files", "description": "Cline-style batch file reader. Read several files (or line ranges) in one call. Accepts a path string, an array of paths, or {files:[{path,start_line,end_line}]} (also paths/file_paths). Line numbers are 1-based and inclusive.", "parameters": {"type": "object", "properties": {"files": {"description": "Path string, array of paths, or array of objects with path and optional 1-based inclusive start_line/end_line."}}, "required": ["files"]}}},
            {"type": "function", "function": {"name": "search_codebase", "description": "Cline-style batch codebase search. Run several grep extended-regex queries at once and return per-query matches with file:line. Prefer this over many separate grep calls.", "parameters": {"type": "object", "properties": {"queries": {"description": "A regex string or array of regex strings to search for."}}, "required": ["queries"]}}},
            {"type": "function", "function": {"name": "run_commands", "description": "Cline-style batch shell runner. Execute several independent bash commands in one call, each with its own exit code, output and duration. Use for parallel investigation; use shell for a single streaming command.", "parameters": {"type": "object", "properties": {"commands": {"description": "A command string, array of command strings, or objects {command,args}."}}, "required": ["commands"]}}},
            {"type": "function", "function": {"name": "fetch_web_content", "description": "Cline-style web fetch. Download one or more URLs and return cleaned readable text. Each request carries a prompt; the returned text plus that prompt is what you analyse, so state what you need extracted.", "parameters": {"type": "object", "properties": {"requests": {"description": "Object {url,prompt} or array of such objects. http:// and https:// only."}}, "required": ["requests"]}}},
            {"type": "function", "function": {"name": "browser", "description": "Real browser automation/crawling (Playwright with automatic Selenium fallback). No manual scripting: pass high-level actions; get structured results. Actions: navigate {url}, fill {selector,value}, click {selector|text|xpath}, extract {selector,mode=text|html|attr|links|title,attr,all,limit}, crawl/rank/search {query,limit,include_text} (extracts & ranks page links for a goal), screenshot {path}, scroll {direction|pixels}, wait {selector|ms}, back, cookies, js {script}. Accepts shorthand action shapes too. Use session for persisted cookies/logins. Use auth_profile for vault-injected session, use_proxy/proxy for interception replay. Use for JS-heavy sites, forms, clicking, crawling, or when fetch_web_content failed. If browser times out, pivot to fetch_web_content/curl and report what was learned.", "parameters": {"type": "object", "properties": {"actions": {"description": "A single action object or an array of action objects, executed in order in one browser session."}, "session": {"type": "string", "description": "Session name for persisted cookies/state (default 'default')."}, "engine": {"type": "string", "enum": ["auto", "playwright", "selenium"], "description": "auto tries Playwright then Selenium; use selenium if playwright is flaky."}, "timeout": {"type": "integer", "description": "Overall timeout in seconds (default 120, clamped 45-180)."}, "auth_profile": {"type": "string", "description": "Vault auth profile name to inject cookies/headers at runtime."}, "use_proxy": {"type": "boolean", "description": "Route through running DarkGPT proxy."}, "proxy": {"type": "string", "description": "Explicit proxy URL, e.g. http://127.0.0.1:8080."}, "headers": {"type": "object"}, "cookies": {"type": "array"}}, "required": ["actions"]}}},
            {"type": "function", "function": {"name": "apply_patch", "description": "Cline-style patch tool. Apply a *** Begin Patch / *** End Patch block supporting *** Update File:, *** Add File:, *** Delete File:, *** Move to: and @@ hunks with context/-/+ lines. Validates every file before writing, then returns per-file diffs. Best choice for multi-file edits.", "parameters": {"type": "object", "properties": {"input": {"type": "string", "description": "The full patch text, including *** Begin Patch and *** End Patch."}}, "required": ["input"]}}},
            {"type": "function", "function": {"name": "editor", "description": "Cline-style single-file editor. Replace old_text with new_text (first occurrence), or insert new_text at insert_line (1-based, inserted before that line; append when beyond EOF). Returns a unified diff.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "new_text": {"type": "string"}, "old_text": {"type": "string"}, "insert_line": {"type": "integer"}}, "required": ["path", "new_text"]}}},
            {"type": "function", "function": {"name": "skills", "description": "Cline-style skill invocation. Load the full instructions of a configured DarkGPT skill by name and follow them for the current task. Use it when a skill matches the work; args may pass a short focus.", "parameters": {"type": "object", "properties": {"skill": {"type": "string", "description": "Skill name (exact, or case-insensitive match)."}, "args": {"type": "string", "description": "Optional focus or arguments."}}, "required": ["skill"]}}},
            {"type": "function", "function": {"name": "ask_question", "description": "Cline-style clarifying question. Ask the user ONE question with 2-5 selectable options when the requirement is genuinely ambiguous. The user's answer is returned to you. Do not use it for questions you can answer by inspecting the workspace.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}, "description": "2-5 short answer choices."}}, "required": ["question", "options"]}}},
            {"type": "function", "function": {"name": "submit_and_exit", "description": "Cline-style completion signal. Call this ONLY when the task is finished and verified. Pass the final summary and whether you verified it; the turn ends immediately with that summary.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "verified": {"type": "boolean"}}, "required": ["summary", "verified"]}}},
            {"type": "function", "function": {"name": "spawn_agent", "description": "Alias of delegate: spawn one isolated read-only subagent (explore/review/test) for a focused parallel task. Returns a concise report.", "parameters": {"type": "object", "properties": {"role": {"type": "string", "enum": ["explore", "review", "test", "recon", "vuln_analysis", "exploit_validation", "reporting"]}, "task": {"type": "string"}, "focus": {"type": "string"}}, "required": ["role", "task"]}}},
            {"type": "function", "function": {"name": "knowledge_graph", "description": "Persist and query a per-target knowledge graph (hosts->services->endpoints->findings/auth/creds/params). Use it to record recon so work resumes across turns. Findings: status attempted|proved|false_positive; proved requires evidence_ref present in evidence ledger. Actions: summary, list_targets, add_host, add_service, add_endpoint, add_finding, link, query, merge.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["summary", "list_targets", "add_host", "add_service", "add_endpoint", "add_finding", "add_credential", "add_auth_state", "add_param", "add_fingerprint", "link", "query", "merge", "export"]}, "target": {"type": "string"}, "host": {"type": "string"}, "port": {"type": "integer"}, "proto": {"type": "string"}, "url": {"type": "string"}, "title": {"type": "string"}, "severity": {"type": "string"}, "status": {"type": "string", "enum": ["attempted", "proved", "false_positive"]}, "details": {"type": "string"}, "evidence_ref": {"type": "string"}, "asset": {"type": "string"}, "repro": {"type": "string"}, "impact": {"type": "string"}, "remediation": {"type": "string"}, "confidence": {"type": "number"}, "name": {"type": "string"}, "kind": {"type": "string"}, "state": {"type": "string"}, "location": {"type": "string"}, "fingerprint": {"type": "string"}, "src_kind": {"type": "string"}, "src_id": {"type": "string"}, "dst_kind": {"type": "string"}, "dst_id": {"type": "string"}, "rel": {"type": "string"}, "text": {"type": "string"}, "limit": {"type": "integer"}, "attrs": {"type": "object"}, "source": {"type": "string"}, "graph": {"type": "object"}, "format": {"type": "string"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "evidence_log", "description": "Append a manual observation to the per-target append-only evidence ledger (JSONL) with SHA-256 and stable evidence id. Tool results are auto-captured when a target is active.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "tool": {"type": "string"}, "args": {}, "result": {"type": "string"}, "ok": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["tool", "result"]}}},
            {"type": "function", "function": {"name": "report_generate", "description": "Generate a pentest report from the target knowledge graph and evidence ledger. Formats: markdown, html, json, csv. JSON includes execution trace; CSV exports findings.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "format": {"type": "string", "enum": ["markdown", "html", "json", "csv"]}, "out": {"type": "string"}}, "required": []}}},
            {"type": "function", "function": {"name": "planner_update", "description": "Update per-target adaptive planner state. Use after probes/observations to record observed signal, hypothesis delta, and next action. Also manage objective, hypotheses, tasks, and dead ends.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "action": {"type": "string", "enum": ["summary", "set_objective", "add_hypothesis", "add_task", "observe", "dead_end"]}, "objective": {"type": "string"}, "text": {"type": "string"}, "title": {"type": "string"}, "status": {"type": "string"}, "owner": {"type": "string"}, "notes": {"type": "string"}, "confidence": {"type": "number"}, "evidence_ref": {"type": "string"}, "tool": {"type": "string"}, "observed": {"type": "string"}, "hypothesis_delta": {"type": "string"}, "next_action": {"type": "string"}, "signature": {"type": "string"}, "reason": {"type": "string"}, "pivot": {"type": "string"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "recon_orchestrate", "description": "Parallel recon: spawns read-only subagents for subdomains, ports/services, tech fingerprinting, endpoints, then merges findings into the target knowledge graph. Requires an active target.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "focus": {"type": "string"}, "workers": {"type": "array", "items": {"type": "string", "enum": ["subdomains", "ports", "tech", "endpoints"]}}}, "required": []}}},
            {"type": "function", "function": {"name": "task_start", "description": "Start a background task (shell command or script) and return immediately with a job id. Use this to fire multiple scripts/tasks in parallel instead of waiting on one terminal. Monitor with task_list/task_wait/task_log/task_kill.", "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["shell", "script"], "description": "shell runs command; script writes+runs python/bash code."}, "command": {"type": "string", "description": "For kind=shell: bash command to run."}, "language": {"type": "string", "enum": ["python", "bash"], "description": "For kind=script."}, "code": {"type": "string", "description": "For kind=script: complete runnable code."}, "args": {"type": "array", "items": {"type": "string"}, "description": "Optional script args."}, "label": {"type": "string", "description": "Short human label."}}, "required": ["kind"]}}},
            {"type": "function", "function": {"name": "task_list", "description": "List background tasks/jobs with status, exit code, duration, and log path.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}}},
            {"type": "function", "function": {"name": "task_wait", "description": "Wait for one or more background task IDs to finish (or timeout). Returns their status and tail output. Use after firing parallel tasks.", "parameters": {"type": "object", "properties": {"job_ids": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "integer", "description": "Seconds to wait, default 30, max 300."}}, "required": ["job_ids"]}}},
            {"type": "function", "function": {"name": "task_log", "description": "Get tail output/log for a background task.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "tail": {"type": "integer"}}, "required": ["job_id"]}}},
            {"type": "function", "function": {"name": "task_kill", "description": "Kill a running background task.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
            {"type": "function", "function": {"name": "http_request", "description": "Make an HTTP request through the DarkGPT harness (not raw model context). Supports named auth profiles from the vault, optional proxy routing, timeout, and response normalization. Raw secrets are injected at runtime and never returned. Use for replay/fuzz/auth-swap without re-crawling.", "parameters": {"type": "object", "properties": {"method": {"type": "string"}, "url": {"type": "string"}, "headers": {"type": "object"}, "params": {"type": "object"}, "data": {}, "json": {}, "auth_profile": {"type": "string", "description": "Named vault profile to inject (headers/cookies/token)."}, "session": {"type": "string"}, "timeout": {"type": "integer"}, "allow_redirects": {"type": "boolean"}, "proxy": {"type": "boolean", "description": "Route through local DarkGPT proxy if running."}, "target": {"type": "string"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "auth_vault", "description": "Manage named auth/session profiles outside LLM context. Actions: list,get,set,delete,inject_preview. get/inject_preview never return raw secrets, only redacted metadata. Use before http_request/browser auth swaps.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["list", "get", "set", "delete", "inject_preview"]}, "target": {"type": "string"}, "name": {"type": "string"}, "profile": {"type": "object", "description": "For set: may include authorization/bearer/token/cookies/cookie/headers/notes. Secrets stored locally, redacted in output."}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "proxy_control", "description": "Control local HTTP interception proxy (mitmdump bridge). Captures flows for replay/fuzz and can import to knowledge graph. Actions: start,stop,status,import,replay. Additive; direct mode still works if proxy unavailable.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["start", "stop", "status", "import", "replay"]}, "target": {"type": "string"}, "port": {"type": "integer"}, "limit": {"type": "integer"}, "url": {"type": "string"}, "auth_profile": {"type": "string"}, "match": {"type": "string", "description": "For replay: substring/url/method filter"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "source_index", "description": "White-box/source-to-sink correlation. Index local source files/directories or fetched JS bundles, extract endpoints/params/sinks, and link them into the target knowledge graph. Use before dynamic probes to turn blind fuzzing into targeted validation.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["index_paths", "fetch_js"]}, "target": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}, "url": {"type": "string"}, "base_url": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "blackboard", "description": "Shared blackboard for multi-agent coordination. Post/read Facts, Intents, and Human Hints; workers can claim/resolve asynchronously. Actions: post,list,claim,resolve,summary.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["post", "list", "claim", "resolve", "summary"]}, "target": {"type": "string"}, "kind": {"type": "string", "enum": ["fact", "intent", "hint"]}, "text": {"type": "string"}, "status": {"type": "string"}, "id": {"type": "string"}, "owner": {"type": "string"}, "note": {"type": "string"}, "refs": {"type": "array", "items": {"type": "string"}}, "confidence": {}, "limit": {"type": "integer"}, "meta": {"type": "object"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "oracle_validate", "description": "Machine oracle validation. Treat scanner/LLM claims as untrusted until deterministic oracle verifies side effect. Kinds: differential_response, timing_delta, reflection_check, oast/interactsh (requires external listener).", "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["differential_response", "timing_delta", "reflection_check", "oast", "interactsh", "callback"]}, "url": {"type": "string"}, "url_a": {"type": "string"}, "url_b": {"type": "string"}, "baseline_url": {"type": "string"}, "marker": {"type": "string"}, "samples": {"type": "integer"}, "headers": {"type": "object"}, "auth_profile": {"type": "string"}, "target": {"type": "string"}}, "required": ["kind"]}}},
            {"type": "function", "function": {"name": "attack_paths", "description": "Rank multi-hop attack paths in the target knowledge graph toward high-value assets. Uses edge provenance and node severity/value heuristics.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "max_paths": {"type": "integer"}, "max_depth": {"type": "integer"}}, "required": []}}},
            {"type": "function", "function": {"name": "verify_findings", "description": "Deterministic replay verification for findings with repro artifacts. Runs stored repro scripts/commands in controlled mode and marks findings still_vulnerable/fixed/error. Actions: list,save_repro,replay,replay_all.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["list", "save_repro", "replay", "replay_all"]}, "target": {"type": "string"}, "finding": {"type": "string", "description": "Finding id/title/key fragment"}, "content": {"type": "string", "description": "For save_repro: script content"}, "kind": {"type": "string", "enum": ["py", "sh"]}, "limit": {"type": "integer"}, "timeout": {"type": "integer"}}, "required": ["action"]}}},
        ]

    def specs(self) -> List[Dict[str, Any]]:
        lsp_on = bool(appconfig.get("lsp_enabled", True))
        scripts_on = bool(appconfig.get("scripts_enabled", True))
        specs = [
            s for s in self.local_specs()
            if s["function"]["name"] not in self.disabled
            and (lsp_on or not s["function"]["name"].startswith("lsp_"))
            and (self.subagents is not None or s["function"]["name"] not in ("delegate", "spawn_agent"))
            and (scripts_on or s["function"]["name"] != "script")
        ]
        if self.mcp:
            specs.extend(
                self.mcp.openai_tools(
                    disabled=self.disabled,
                    disabled_servers=self.disabled_servers,
                )
            )
        return specs

    @staticmethod
    def display_name(name: str) -> str:
        if name.startswith("mcp__"):
            parts = name.split("__", 2)
            if len(parts) == 3:
                return f"{parts[1]}:{parts[2]}"
        return name

    async def run(
        self,
        name: str,
        args: Dict[str, Any],
        emit: Optional[Callable[[str, str], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Gate every tool through hooks + permissions, then dispatch it."""
        if name == "ask_question":
            question = str(args.get("question") or "").strip()
            options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()][:5]
            answer = ""
            if self.ask_callback is not None:
                try:
                    answer = str(await self.ask_callback(question, options) or "")
                except Exception:
                    answer = ""
            return clamp(json.dumps({
                "question": question,
                "options": options,
                "answer": answer,
                "answered": bool(answer),
                "note": "" if answer else "No answer supplied; ask the user in your final reply.",
            }, ensure_ascii=False))
        if name == "spawn_agent":
            name = "delegate"
        cwd = os.getcwd()
        risk = permissions.risk_of(name, args)
        ctx = {
            "tool": name,
            "path": str(args.get("path") or ""),
            "command": str(args.get("command") or args.get("task") or args.get("code") or ""),
        }
        hook_notes: List[str] = []
        if self.hooks_enabled:
            for event in ("pre_tool", _pre_event(risk)):
                outcome = await asyncio.to_thread(hooks.run, event, ctx, cwd)
                if not outcome.allowed:
                    if on_event:
                        on_event({"type": "hook_blocked", "event": event, "reason": outcome.reason()})
                    return json.dumps({"error": f"blocked by {event} hook: {outcome.reason()}", "blocked": True}, ensure_ascii=False)
                hook_notes.extend(outcome.outputs)

        prompt, _, sig = permissions.needs_prompt(name, args, self.permission_mode, self.approved)
        if prompt:
            allowed = False
            if self.permission_callback is not None:
                try:
                    allowed = bool(await self.permission_callback(name, args, risk))
                except Exception:
                    allowed = False
            else:
                allowed = permissions.default_allow(name, args, self.permission_mode)
            if not allowed:
                if on_event:
                    on_event({"type": "permission_denied", "tool": name, "risk": risk})
                return json.dumps({"error": f"permission denied for {name} ({risk})", "denied": True}, ensure_ascii=False)
            self.approved.add(sig)

        result = await self._dispatch(name, args, emit=emit, on_event=on_event)

        if self.hooks_enabled:
            for event in (_post_event(risk), "post_tool"):
                outcome = await asyncio.to_thread(hooks.run, event, ctx, cwd)
                hook_notes.extend(outcome.outputs)
        if hook_notes:
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    data["hook_output"] = " | ".join(hook_notes)[:2000]
                    result = clamp(json.dumps(data, ensure_ascii=False))
            except Exception:
                pass
        return result

    async def _dispatch(
        self,
        name: str,
        args: Dict[str, Any],
        emit: Optional[Callable[[str, str], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        try:
            if name == "delegate":
                if not self.subagents:
                    return json.dumps({"error": "subagents are disabled"})
                role = str(args.get("role") or "explore")
                task = str(args.get("task") or "")
                focus = str(args.get("focus") or "")
                if on_event:
                    on_event({"type": "subagent_start", "role": role, "task": task})
                result = await self.subagents.run(role, task, focus)
                try:
                    _target = str(appconfig.get("active_target") or "").strip()
                    if _target and result.ok:
                        await asyncio.to_thread(self._merge_subagent_into_target, _target, result)
                except Exception:
                    pass
                if on_event:
                    on_event({
                        "type": "subagent_end", "role": result.role, "task": result.task,
                        "ok": result.ok, "duration": result.duration,
                        "preview": result.text[:240], "error": result.error,
                    })
                return clamp(json.dumps(result.to_dict(), ensure_ascii=False))
            if name == "knowledge_graph":
                return clamp(json.dumps(await asyncio.to_thread(self._run_kg, args), ensure_ascii=False))
            if name == "evidence_log":
                return clamp(json.dumps(await asyncio.to_thread(self._run_evidence_log, args), ensure_ascii=False))
            if name == "report_generate":
                return clamp(json.dumps(await asyncio.to_thread(self._run_report, args), ensure_ascii=False))
            if name == "planner_update":
                return clamp(json.dumps(await asyncio.to_thread(self._run_planner, args), ensure_ascii=False))
            if name == "recon_orchestrate":
                return clamp(json.dumps(await self._run_recon_orchestrate(args, on_event=on_event), ensure_ascii=False))
            if name == "task_start":
                kind = str(args.get("kind") or "shell").strip().lower()
                label = str(args.get("label") or "")
                cwd = os.getcwd()
                if kind == "script":
                    return clamp(json.dumps(await self.jobs.start_script(
                        str(args.get("language") or "python"),
                        str(args.get("code") or ""),
                        list(args.get("args") or []),
                        cwd=cwd,
                        label=label,
                    ), ensure_ascii=False))
                return clamp(json.dumps(await self.jobs.start_shell(str(args.get("command") or ""), cwd=cwd, label=label), ensure_ascii=False))
            if name == "task_list":
                return clamp(json.dumps(self.jobs.list(limit=int(args.get("limit", 20))), ensure_ascii=False))
            if name == "task_wait":
                ids = args.get("job_ids") or []
                if isinstance(ids, str): ids = [ids]
                return clamp(json.dumps(await self.jobs.wait([str(i) for i in ids], timeout=int(args.get("timeout", 30))), ensure_ascii=False))
            if name == "task_log":
                return clamp(json.dumps(self.jobs.log(str(args.get("job_id") or ""), tail=int(args.get("tail", 4000))), ensure_ascii=False))
            if name == "task_kill":
                return clamp(json.dumps(await self.jobs.kill(str(args.get("job_id") or "")), ensure_ascii=False))
            if name == "http_request":
                return clamp(json.dumps(await asyncio.to_thread(self._run_http_request, args), ensure_ascii=False))
            if name == "auth_vault":
                return clamp(json.dumps(await asyncio.to_thread(self._run_auth_vault, args), ensure_ascii=False))
            if name == "proxy_control":
                return clamp(json.dumps(await self._run_proxy_control(args), ensure_ascii=False))
            if name == "source_index":
                return clamp(json.dumps(await asyncio.to_thread(self._run_source_index, args), ensure_ascii=False))
            if name == "blackboard":
                return clamp(json.dumps(await asyncio.to_thread(self._run_blackboard, args), ensure_ascii=False))
            if name == "oracle_validate":
                return clamp(json.dumps(await asyncio.to_thread(self._run_oracle, args), ensure_ascii=False))
            if name == "attack_paths":
                return clamp(json.dumps(await asyncio.to_thread(self._run_attack_paths, args), ensure_ascii=False))
            if name == "verify_findings":
                return clamp(json.dumps(await self._run_verify_findings(args), ensure_ascii=False))
            if name.startswith("mcp__"):
                _, server, tool = name.split("__", 2)
                if not self.mcp:
                    return json.dumps({"error": "mcp not available"})
                res = await self.mcp.call_tool(server, tool, args)
                content = getattr(res, "content", res)
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        txt = getattr(c, "text", None)
                        parts.append(txt if txt is not None else str(c))
                    return clamp("\n".join(parts))
                return clamp(json.dumps(content, ensure_ascii=False, default=str))
            if name == "shell" and emit is not None:
                return await self._run_shell_stream(args, emit)
            if name == "script" and emit is not None:
                return await self._run_script_stream(args, emit)
            undo_id = ""
            if name in EDIT_TOOLS and (name != "lsp_rename" or args.get("apply")):
                paths = []
                if name == "rename":
                    preview = codeintel.rename(
                        str(args.get("symbol") or ""), str(args.get("new_name") or ""),
                        os.getcwd(), str(args.get("path") or ""), True,
                    )
                    paths = [item["path"] for item in preview.get("files", [])]
                elif name == "lsp_rename" and args.get("apply"):
                    preview = lsp.rename_symbol(
                        str(args.get("path") or ""), str(args.get("symbol") or ""),
                        str(args.get("new_name") or ""), os.getcwd(),
                        int(args.get("line", 0)), int(args.get("character", 0)), False,
                    )
                    paths = [lsp._path_from_uri(uri) for uri in (preview.get("changes") or {})]
                    paths = [p for p in paths if p]
                elif name == "apply_patch":
                    paths = [op["path"] for op in cline_tools._parse_patch(str(args.get("input") or "")) if op.get("path")]
                elif args.get("path"):
                    paths = [str(args["path"])]
                undo_id = await asyncio.to_thread(undo.save_paths, os.getcwd(), paths)
            result = clamp(json.dumps(await asyncio.to_thread(self._run_local, name, args), ensure_ascii=False))
            if undo_id:
                try:
                    data = json.loads(result)
                    if isinstance(data, dict):
                        data["undo_id"] = undo_id
                        result = clamp(json.dumps(data, ensure_ascii=False))
                except Exception:
                    pass
            return result
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    async def _run_shell_stream(self, args: Dict[str, Any], emit: Callable[[str, str], None]) -> str:
        command = args.get("command", "")
        timeout = min(int(args.get("timeout", 120)), 300)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
        )
        chunks: List[str] = []

        async def pump(stream: Optional[asyncio.StreamReader], label: str) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace")
                chunks.append(text)
                emit(label, text)

        try:
            await asyncio.wait_for(
                asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr")),
                timeout=timeout,
            )
            code = await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            chunks.append(f"\n[timed out after {timeout}s]")
            code = -1
        out = "".join(chunks).strip()
        return clamp(json.dumps({"exit_code": code, "output": out}, ensure_ascii=False))

    async def _run_script_stream(self, args: Dict[str, Any], emit: Callable[[str, str], None]) -> str:
        """Author a helper script, run it with live output, keep the file."""
        started = time.perf_counter()
        try:
            path, command = await asyncio.to_thread(
                scripting.build_argv,
                str(args.get("language") or "python"),
                str(args.get("code") or ""),
                list(args.get("args") or []),
            )
        except scripting.ScriptError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        timeout = max(1, min(int(args.get("timeout", scripting.DEFAULT_TIMEOUT)), scripting.MAX_TIMEOUT))
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=os.getcwd(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"cannot run script: {exc}", "script_path": str(path)}, ensure_ascii=False)

        chunks: List[str] = []

        async def pump(stream: Optional[asyncio.StreamReader], label: str) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace")
                chunks.append(text)
                emit(label, text)

        try:
            await asyncio.wait_for(
                asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr")),
                timeout=timeout,
            )
            code = await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            chunks.append(f"\n[timed out after {timeout}s]")
            code = -1
        output = "".join(chunks).strip()
        if len(output) > scripting.MAX_OUTPUT:
            output = output[:scripting.MAX_OUTPUT] + f"\n...[truncated {len(output) - scripting.MAX_OUTPUT} chars]"
        if not output:
            output = "(no output)"
        return clamp(json.dumps({
            "ok": code == 0,
            "exit_code": code,
            "output": output,
            "script_path": str(path),
            "language": scripting.normalise_language(str(args.get("language") or "python")),
            "duration": max(0.0, time.perf_counter() - started),
        }, ensure_ascii=False))

    def _active_target(self, args: Dict[str, Any]) -> str:
        t = str((args or {}).get("target") or "").strip()
        if t:
            return t
        return str(appconfig.get("active_target") or "").strip()

    def _merge_subagent_into_target(self, target: str, result: Any) -> Dict[str, int]:
        """Merge structured subagent output into KG with light dedupe and provenance."""
        parsed = getattr(result, "parsed", None) or {}
        role = getattr(result, "role", "")
        imported = {"hosts": 0, "services": 0, "endpoints": 0, "findings": 0, "intents": 0}
        if not isinstance(parsed, dict):
            return imported
        try:
            from agent import blackboard as bb
        except Exception:
            bb = None
        def _dedupe(rows: Any, keys: List[str]) -> List[Dict[str, Any]]:
            seen = set(); out = []
            for r in rows or []:
                if not isinstance(r, dict): continue
                sig = "|".join(str(r.get(k, "")).strip().lower() for k in keys)
                if not sig or sig in seen: continue
                seen.add(sig); out.append(r)
            return out
        for h in _dedupe(parsed.get("hosts"), ["host"]):
            host = str(h.get("host") or "").strip()
            if host:
                kg_util.add_host(target, host, {"notes": h.get("notes", "")}, source=f"subagent:{role}")
                imported["hosts"] += 1
        for srow in _dedupe(parsed.get("services"), ["host", "port", "proto"]):
            host = str(srow.get("host") or "").strip()
            port = srow.get("port")
            if host and port:
                kg_util.add_service(target, host, port, str(srow.get("proto") or "tcp"), {"name": srow.get("name", ""), "notes": srow.get("notes", "")}, source=f"subagent:{role}")
                imported["services"] += 1
        for e in _dedupe(parsed.get("endpoints"), ["url"]):
            url = str(e.get("url") or "").strip()
            if url:
                kg_util.add_endpoint(target, url, {"notes": e.get("notes", ""), "method": e.get("method", ""), "auth": e.get("auth", "")}, source=f"subagent:{role}")
                imported["endpoints"] += 1
                if bb:
                    try:
                        bb.post(target, "intent", f"Probe endpoint {url} ({e.get('method','')}) from {role}", source=f"subagent:{role}", refs=[url])
                        imported["intents"] += 1
                    except Exception:
                        pass
        findings_src = parsed.get("findings") or parsed.get("validations") or []
        for f in _dedupe(findings_src, ["title", "asset", "severity"]):
            title = str(f.get("title") or "").strip()
            if not title: continue
            attrs = {
                "status": f.get("status", "attempted"),
                "asset": f.get("asset", ""),
                "repro": f.get("repro") or f.get("reproduction") or f.get("minimal_test") or "",
                "impact": f.get("impact", ""),
                "remediation": f.get("remediation", ""),
                "confidence": f.get("confidence"),
                "oracle_ref": f.get("oracle_ref", ""),
                "oracle_verified": bool(f.get("oracle_verified")),
            }
            node = kg_util.add_finding(target, title, str(f.get("severity") or "info"), str(f.get("details") or f.get("why") or ""), str(f.get("evidence_ref") or ""), attrs, source=f"subagent:{role}")
            imported["findings"] += 1
            if bb:
                try:
                    bb.post(target, "fact", f"Finding {title} [{attrs.get('status')}] {attrs.get('asset','')}", source=f"subagent:{role}", refs=[str(node.get('key') or '')])
                except Exception:
                    pass
        next_step = str(parsed.get("next_step") or "").strip()
        if next_step and bb:
            try:
                bb.post(target, "intent", next_step[:500], source=f"subagent:{role}")
                imported["intents"] += 1
            except Exception:
                pass
        if any(imported.values()):
            evidence_util.record(target, tool=f"subagent_merge:{role}", args={"task": str(getattr(result,'task',''))[:200]}, result=json.dumps(imported), ok=True, source="agent", tags=["subagent","merge",role])
        return imported

    def _run_kg(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        action = str(a.get("action") or "").strip()
        target = self._active_target(a)
        if action == "list_targets":
            return {"targets": kg_util.list_targets(int(a.get("limit", 50)))}
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        if action == "summary":
            return kg_util.summary(target)
        if action == "add_host":
            return kg_util.add_host(target, str(a.get("host") or ""), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "add_service":
            return kg_util.add_service(target, str(a.get("host") or ""), a.get("port", 0), str(a.get("proto") or "tcp"), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "add_endpoint":
            return kg_util.add_endpoint(target, str(a.get("url") or ""), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "add_finding":
            attrs = dict(a.get("attrs") or {})
            for k in ("status", "asset", "repro", "reproduction", "impact", "remediation", "confidence"):
                if a.get(k) is not None and a.get(k) != "":
                    attrs.setdefault(k, a.get(k))
            return kg_util.add_finding(target, str(a.get("title") or "Finding"), str(a.get("severity") or "info"), str(a.get("details") or ""), str(a.get("evidence_ref") or ""), attrs, source=str(a.get("source") or "agent"))
        if action == "add_credential":
            return kg_util.add_credential(target, str(a.get("name") or a.get("title") or "cred"), str(a.get("kind") or "token"), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "add_auth_state":
            return kg_util.add_auth_state(target, str(a.get("host") or a.get("asset") or ""), str(a.get("state") or a.get("status") or "unknown"), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "add_param":
            return kg_util.add_param(target, str(a.get("url") or a.get("endpoint") or ""), str(a.get("name") or ""), str(a.get("location") or "query"), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "add_fingerprint":
            return kg_util.add_fingerprint(target, str(a.get("asset") or a.get("url") or a.get("host") or ""), str(a.get("fingerprint") or a.get("details") or ""), a.get("attrs") or {}, source=str(a.get("source") or "agent"))
        if action == "link":
            ok = kg_util.link(target, str(a.get("src_kind") or ""), str(a.get("src_id") or ""), str(a.get("dst_kind") or ""), str(a.get("dst_id") or ""), str(a.get("rel") or "related"))
            return {"ok": ok}
        if action == "query":
            return {"nodes": kg_util.query(target, str(a.get("kind") or ""), str(a.get("text") or ""), int(a.get("limit", 50)))}
        if action == "merge":
            return kg_util.merge(target, a.get("graph") or {}, source=str(a.get("source") or "merge"))
        if action == "export":
            return kg_util.export_graph(target, str(a.get("format") or "json"))
        return {"error": f"unknown knowledge_graph action: {action}"}

    def _run_evidence_log(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        return evidence_util.record(target, tool=str(a.get("tool") or "manual"), args=a.get("args"), result=str(a.get("result") or ""), ok=bool(a.get("ok", True)), source="manual", tags=a.get("tags") or [])

    def _run_report(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        return reporting_util.write_report(target, fmt=str(a.get("format") or "markdown"), out=str(a.get("out") or ""))

    def _run_planner(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        action = str(a.get("action") or "summary").strip()
        if action == "summary":
            return planner_util.summary(target)
        if action == "set_objective":
            planner_util.set_objective(target, str(a.get("objective") or a.get("text") or ""))
            return planner_util.summary(target)
        if action == "add_hypothesis":
            planner_util.add_hypothesis(target, str(a.get("text") or a.get("title") or ""), str(a.get("status") or "open"), a.get("confidence"), str(a.get("evidence_ref") or ""))
            return planner_util.summary(target)
        if action == "add_task":
            planner_util.add_task(target, str(a.get("title") or a.get("text") or ""), str(a.get("status") or "todo"), str(a.get("owner") or ""), str(a.get("notes") or ""))
            return planner_util.summary(target)
        if action == "observe":
            planner_util.record_observation(target, tool=str(a.get("tool") or "manual"), observed=str(a.get("observed") or a.get("text") or ""), hypothesis_delta=str(a.get("hypothesis_delta") or ""), next_action=str(a.get("next_action") or ""))
            return planner_util.summary(target)
        if action == "dead_end":
            planner_util.record_dead_end(target, str(a.get("signature") or a.get("text") or ""), str(a.get("reason") or ""), str(a.get("pivot") or a.get("next_action") or ""))
            return planner_util.summary(target)
        return {"error": f"unknown planner action: {action}"}

    async def _run_recon_orchestrate(self, args: Dict[str, Any], on_event=None) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        focus = str(a.get("focus") or "").strip()
        workers = a.get("workers") or ["subdomains", "ports", "tech", "endpoints"]
        workers = [str(w) for w in workers if str(w) in {"subdomains", "ports", "tech", "endpoints"}] or ["subdomains", "ports", "tech", "endpoints"]
        pool = self.subagents or subagents.SubagentPool(self.model if hasattr(self, "model") else str(appconfig.get("model") or "nvidia/nemotron-3.5-lightning-30b-a3b"), os.getcwd(), max_concurrent=2, max_total=max(3, len(workers)))
        prompts = {
            "subdomains": f"Enumerate subdomains and related hosts for target '{target}'. Use read-only tools. Return JSON schema for recon role (hosts/services/endpoints/findings/next_step).",
            "ports": f"Identify open ports/services for '{target}' from safe probes available in shell. Return JSON schema for recon role.",
            "tech": f"Fingerprint web technologies/endpoints/auth surfaces for '{target}' (headers, HTML clues, common paths). Return JSON schema for recon role.",
            "endpoints": f"Discover interesting endpoints/paths for '{target}' (robots, sitemap, js links, common API paths). Return JSON schema for recon role.",
        }
        tasks = [("recon", prompts[w] + (f"\nFocus: {focus}" if focus else ""), f"Target: {target}") for w in workers]
        results = await pool.run_many(tasks)
        imported = {"hosts": 0, "services": 0, "endpoints": 0, "findings": 0}
        for r in results:
            merged = self._merge_subagent_into_target(target, r)
            for k in imported:
                imported[k] += int(merged.get(k, 0))
            # fallback line parsing if nothing structured merged
            if not any(merged.values()):
                text = r.text or ""
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("HOST "):
                        parts = line.split(None, 2)
                        if len(parts) >= 2:
                            kg_util.add_host(target, parts[1], {"notes": parts[2] if len(parts) > 2 else ""}, source="recon_orchestrate")
                            imported["hosts"] += 1
                    elif line.startswith("SERVICE "):
                        parts = line.split(None, 4)
                        if len(parts) >= 4:
                            try:
                                kg_util.add_service(target, parts[1], int(parts[2]), parts[3], {"name": parts[4] if len(parts) > 4 else ""}, source="recon_orchestrate")
                                imported["services"] += 1
                            except Exception:
                                pass
                    elif line.startswith("ENDPOINT "):
                        parts = line.split(None, 2)
                        if len(parts) >= 2:
                            kg_util.add_endpoint(target, parts[1], {"notes": parts[2] if len(parts) > 2 else ""}, source="recon_orchestrate")
                            imported["endpoints"] += 1
                    elif line.startswith("FINDING "):
                        rest = line[len("FINDING "):]
                        sev, _, tail = rest.partition(" ")
                        title, _, details = tail.partition("::")
                        kg_util.add_finding(target, title.strip() or "Recon finding", sev.strip().lower() or "info", details.strip(), source="recon_orchestrate")
                        imported["findings"] += 1
            evidence_util.record(target, tool=f"recon_orchestrate:{r.role}", args={"task": r.task[:200]}, result=r.text or r.error, ok=r.ok, source="recon_orchestrate", tags=["recon", r.role])
        return {"target": target, "workers": workers, "results": [r.to_dict() for r in results], "imported": imported, "graph": kg_util.summary(target)}

    def _run_auth_vault(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        action = str(a.get("action") or "list").strip()
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        name = str(a.get("name") or a.get("profile_name") or "")
        if action == "list":
            return auth_vault.list_profiles(target)
        if action == "get":
            return auth_vault.get_profile(target, name, include_secrets=False)
        if action == "set":
            profile = a.get("profile") if isinstance(a.get("profile"), dict) else {k: v for k, v in a.items() if k not in {"action", "target", "name", "profile_name"}}
            return auth_vault.set_profile(target, name, profile)
        if action == "delete":
            return auth_vault.delete_profile(target, name)
        if action == "inject_preview":
            # redacted preview only
            res = auth_vault.get_profile(target, name, include_secrets=False)
            if not res.get("ok"):
                return res
            prof = res.get("profile") or {}
            keys = [k for k in prof.keys() if k not in ("name","updated_at")]
            return {"ok": True, "target": target, "profile": name, "injectable_keys": keys, "note": "Secrets are injected at runtime only; raw values never returned."}
        return {"error": f"unknown auth_vault action: {action}"}

    def _run_http_request(self, args: Dict[str, Any]) -> Any:
        import requests as _rq
        a = args or {}
        target = self._active_target(a)
        method = str(a.get("method") or "GET").upper().strip()
        url = str(a.get("url") or "").strip()
        if not url:
            return {"error": "url required"}
        headers = dict(a.get("headers") or {})
        auth_profile = str(a.get("auth_profile") or a.get("profile") or "").strip()
        if target and auth_profile:
            headers = auth_vault.inject_headers(target, auth_profile, headers)
        timeout = max(1, min(int(a.get("timeout", 30)), 120))
        kwargs: Dict[str, Any] = {
            "headers": headers,
            "params": a.get("params") or None,
            "timeout": timeout,
            "allow_redirects": bool(a.get("allow_redirects", True)),
        }
        if a.get("json") is not None:
            kwargs["json"] = a.get("json")
        elif a.get("data") is not None:
            kwargs["data"] = a.get("data")
        # proxy routing
        use_proxy = bool(a.get("proxy"))
        proxy_url = ""
        if use_proxy and getattr(self, "proxy", None) and self.proxy.status().get("running"):
            port = self.proxy.status().get("port")
            proxy_url = f"http://127.0.0.1:{port}"
            kwargs["proxies"] = {"http": proxy_url, "https": proxy_url}
        try:
            resp = _rq.request(method, url, **kwargs)
            body = resp.text or ""
            preview = body[:4000]
            parsed_headers = {k: ("[REDACTED]" if k.lower() in {"set-cookie","authorization","cookie"} else v) for k, v in resp.headers.items()}
            out = {
                "ok": True,
                "status": resp.status_code,
                "url": str(resp.url),
                "method": method,
                "headers": parsed_headers,
                "body_preview": preview,
                "body_bytes": len(resp.content or b""),
                "elapsed_ms": int(getattr(resp, "elapsed", None).total_seconds() * 1000) if getattr(resp, "elapsed", None) else None,
                "auth_profile": auth_profile or None,
                "proxy": proxy_url or None,
            }
            # auto-upsert endpoint/params to KG
            if target and str(resp.url).startswith(("http://","https://")):
                try:
                    kg_util.add_endpoint(target, str(resp.url).split("?")[0], {"method": method, "status": resp.status_code, "seen_via": "http_request"}, source="http_request")
                    from urllib.parse import urlparse, parse_qs
                    for pn in parse_qs(urlparse(str(resp.url)).query).keys():
                        kg_util.add_param(target, str(resp.url).split("?")[0], pn, "query", {"seen_via": "http_request"}, source="http_request")
                except Exception:
                    pass
            return out
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url, "method": method, "auth_profile": auth_profile or None, "proxy": proxy_url or None}

    async def _run_proxy_control(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        action = str(a.get("action") or "status").strip()
        target = self._active_target(a) or "default"
        if action == "start":
            return await self.proxy.start(target, port=int(a.get("port", 8080)))
        if action == "stop":
            return await self.proxy.stop()
        if action == "status":
            return self.proxy.status()
        if action == "import":
            return self.proxy.export_to_graph(target=target, limit=int(a.get("limit", 500)))
        if action == "replay":
            # replay captured flows matching url/match via http_request harness
            match = str(a.get("match") or a.get("url") or "").lower()
            lp = proxy_util.PROXY_ROOT / kg_util.slugify(target) / "flows.jsonl"
            if not lp.exists():
                return {"ok": False, "error": f"no flow log: {lp}"}
            flows = []
            for line in lp.read_text(encoding="utf-8", errors="replace").splitlines()[-int(a.get("limit", 200)):]:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                url = str(e.get("url") or "")
                if match and match not in url.lower() and match != str(e.get("method","")).lower():
                    continue
                flows.append(e)
            results = []
            for e in flows[:20]:
                req = {"url": e.get("url"), "method": e.get("method") or "GET", "target": target}
                if a.get("auth_profile"):
                    req["auth_profile"] = a.get("auth_profile")
                results.append(await asyncio.to_thread(self._run_http_request, req))
            return {"ok": True, "target": target, "matched": len(flows), "replayed": len(results), "results": results}
        return {"error": f"unknown proxy_control action: {action}"}

    def _run_source_index(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        action = str(a.get("action") or "index_paths").strip()
        if action == "index_paths":
            paths = a.get("paths") or []
            if isinstance(paths, str): paths = [paths]
            return source_correlation.index_paths(target, [str(p) for p in paths], base_url=str(a.get("base_url") or ""), limit=int(a.get("limit", 40)))
        if action == "fetch_js":
            return source_correlation.fetch_and_index_js(target, str(a.get("url") or ""), timeout=int(a.get("timeout", 20)))
        return {"error": f"unknown source_index action: {action}"}

    def _run_blackboard(self, args: Dict[str, Any]) -> Any:
        from agent import blackboard as bb
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        action = str(a.get("action") or "summary").strip()
        if action == "post":
            item = bb.post(target, str(a.get("kind") or "intent"), str(a.get("text") or ""), status=str(a.get("status") or "open"), source=str(a.get("source") or "agent"), refs=a.get("refs"), confidence=a.get("confidence"), meta=a.get("meta"))
            return {"ok": True, "item": item}
        if action == "list":
            return {"ok": True, "items": bb.list_items(target, kind=str(a.get("kind") or ""), status=str(a.get("status") or ""), limit=int(a.get("limit", 50)))}
        if action == "claim":
            return bb.claim(target, str(a.get("id") or ""), str(a.get("owner") or "agent"))
        if action == "resolve":
            return bb.resolve(target, str(a.get("id") or ""), str(a.get("status") or "done"), str(a.get("note") or ""))
        if action == "summary":
            return {"ok": True, **bb.summary(target)}
        return {"error": f"unknown blackboard action: {action}"}

    def _run_oracle(self, args: Dict[str, Any]) -> Any:
        from agent import oracle
        a = dict(args or {})
        target = self._active_target(a)
        kind = str(a.get("kind") or "")
        # inject auth profile headers if provided
        headers = dict(a.get("headers") or {})
        prof = str(a.get("auth_profile") or "")
        if target and prof:
            headers = auth_vault.inject_headers(target, prof, headers)
        a["headers"] = headers
        res = oracle.run_oracle(kind, **a)
        # log oracle verdict as evidence/blackboard fact
        if target:
            try:
                evidence_util.record(target, tool="oracle_validate", args={"kind": kind, "url": a.get("url") or a.get("url_a")}, result=json.dumps(res, ensure_ascii=False)[:4000], ok=bool(res.get("verdict") and res.get("verdict") != "external_listener_required"), source="oracle", tags=["oracle", kind])
            except Exception:
                pass
        return res

    def _run_attack_paths(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        return {"target": target, "paths": kg_util.rank_attack_paths(target, max_paths=int(a.get("max_paths", 10)), max_depth=int(a.get("max_depth", 5)))}

    async def _run_verify_findings(self, args: Dict[str, Any]) -> Any:
        a = args or {}
        target = self._active_target(a)
        if not target:
            return {"error": "no target set; pass target or run /target <name>"}
        action = str(a.get("action") or "list").strip()
        findings = kg_util.query(target, kind="finding", limit=int(a.get("limit", 200)))
        needle = str(a.get("finding") or "").lower().strip()
        if needle:
            findings = [f for f in findings if needle in str(f.get("id","")).lower() or needle in str(f.get("key","")).lower() or needle in str((f.get("attrs") or {}).get("title","")).lower()]
        if action == "list":
            out = []
            for f in findings:
                attrs = f.get("attrs") or {}
                out.append({"id": f.get("id"), "key": f.get("key"), "title": attrs.get("title"), "status": attrs.get("status"), "severity": attrs.get("severity"), "repro_script": attrs.get("repro_script"), "repro_command": attrs.get("repro_command"), "evidence_ref": attrs.get("evidence_ref"), "last_replay_verdict": attrs.get("last_replay_verdict")})
            return {"ok": True, "target": target, "findings": out}
        if action == "save_repro":
            if not findings:
                return {"ok": False, "error": "finding not found"}
            f = findings[0]
            attrs = dict(f.get("attrs") or {})
            content = str(a.get("content") or "")
            kind = str(a.get("kind") or ("sh" if str(a.get("filename","")).endswith(".sh") else "py"))
            if not content.strip():
                return {"ok": False, "error": "content required"}
            art = evidence_util.save_repro_artifact(target, str(f.get("id") or attrs.get("title") or "finding"), content, kind=kind)
            if art.get("ok"):
                attrs["repro_script"] = art.get("path")
                attrs["repro_sha256"] = art.get("sha256")
                kg_util.upsert_node(target, "finding", str(attrs.get("title") or f.get("id")), attrs, source="verify_findings")
                evidence_util.record(target, tool="verify_findings", args={"action":"save_repro","finding":f.get("id")}, result=json.dumps(art), ok=True, source="verify_findings", tags=["repro","artifact"])
            return art
        if action in ("replay", "replay_all"):
            selected = findings if action == "replay_all" else findings[:1]
            results = []
            for f in selected:
                attrs = dict(f.get("attrs") or {})
                repro_script = str(attrs.get("repro_script") or "")
                repro_cmd = str(attrs.get("repro_command") or "")
                if not repro_script and not repro_cmd:
                    results.append({"finding": f.get("id"), "status": "skipped", "reason": "no repro_script/repro_command"})
                    continue
                if repro_script:
                    # run script path with bash/python based on suffix
                    p = Path(repro_script).expanduser()
                    if not p.is_absolute(): p = (kg_util.target_dir(target) / repro_script).resolve()
                    if not p.exists():
                        results.append({"finding": f.get("id"), "status": "error", "reason": f"missing repro script: {p}"})
                        continue
                    cmd = ["/bin/bash", str(p)] if p.suffix == ".sh" else [sys.executable, str(p)]
                else:
                    cmd = ["/bin/bash", "-lc", repro_cmd]
                try:
                    proc = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                    out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=max(10, min(int(a.get("timeout", 120)), 600)))
                    out = out_b.decode("utf-8", errors="replace")[-8000:]
                    exit_code = proc.returncode
                except Exception as exc:
                    out = str(exc); exit_code = -1
                verdict = "still_vulnerable" if exit_code == 0 else "fixed_or_not_reproduced"
                # update finding node status attrs
                attrs["last_replay_exit"] = exit_code
                attrs["last_replay_verdict"] = verdict
                attrs["last_replay_at"] = time.time()
                attrs["last_replay_output"] = out[:2000]
                if verdict == "still_vulnerable":
                    attrs["status"] = "proved"
                kg_util.upsert_node(target, "finding", str(attrs.get("title") or f.get("id")), attrs, source="verify_findings")
                evidence_util.record(target, tool="verify_findings", args={"finding": f.get("id")}, result=out, ok=exit_code==0, source="verify_findings", tags=["replay", verdict])
                results.append({"finding": f.get("id"), "title": attrs.get("title"), "exit_code": exit_code, "verdict": verdict, "output": out[:2000]})
            return {"ok": True, "target": target, "results": results}
        return {"error": f"unknown verify_findings action: {action}"}

    def _run_local(self, name: str, args: Dict[str, Any]) -> Any:
        a = args or {}
        cwd = os.getcwd()
        if name == "shell":
            p = subprocess.run(a.get("command", ""), shell=True, cwd=os.getcwd(), capture_output=True, text=True, timeout=min(int(a.get("timeout", 120)), 300), executable="/bin/bash")
            out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
            return {"exit_code": p.returncode, "output": clamp(out.strip())}
        if name == "read":
            pth = safe_path(a["path"])
            if not pth.exists(): return {"error": f"not found: {pth}"}
            lines = pth.read_text(errors="replace").splitlines()
            off = max(int(a.get("offset", 1)) - 1, 0)
            lim = int(a.get("limit", 400))
            chunk = [f"{off+i+1}: {l}" for i, l in enumerate(lines[off:off+lim])]
            return {"path": str(pth), "content": clamp("\n".join(chunk)), "total_lines": len(lines)}
        if name == "write":
            pth = safe_path(a["path"]); pth.parent.mkdir(parents=True, exist_ok=True)
            before = pth.read_text(errors="replace") if pth.exists() else ""
            after = a.get("content", "")
            pth.write_text(after)
            diff = "".join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                fromfile=str(pth), tofile=str(pth), n=3,
            ))
            return {"path": str(pth), "ok": True, "diff": clamp(diff, 6000)}
        if name == "edit":
            pth = safe_path(a["path"])
            if not pth.exists(): return {"error": f"not found: {pth}"}
            txt = pth.read_text(errors="replace")
            old = a.get("old", ""); new = a.get("new", "")
            expanded = False
            if old and old not in txt:
                # Accept a line range ("55-64" or "55") instead of exact text.
                m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", old)
                if m:
                    lines = txt.split("\n")
                    start = int(m.group(1))
                    end = int(m.group(2) or start)
                    if 1 <= start <= end <= len(lines):
                        old = "\n".join(lines[start - 1:end])
                        expanded = True
            if not old or old not in txt:
                return {"error": "old text not found (use exact text or a line range like '55-64')"}
            after = txt.replace(old, new, 1)
            pth.write_text(after)
            diff = "".join(difflib.unified_diff(
                txt.splitlines(True), after.splitlines(True),
                fromfile=str(pth), tofile=str(pth), n=3,
            ))
            return {"path": str(pth), "ok": True, "line_range": expanded, "diff": clamp(diff, 6000)}
        if name == "ls":
            pth = safe_path(a.get("path", "."))
            if not pth.exists(): return {"error": f"not found: {pth}"}
            if pth.is_file(): return {"path": str(pth), "type": "file"}
            entries = [{"name": c.name, "type": "dir" if c.is_dir() else "file"} for c in sorted(pth.iterdir(), key=lambda x:(not x.is_dir(), x.name.lower()))[:200]]
            return {"path": str(pth), "entries": entries}
        if name == "grep":
            cmd = ["grep", "-RIn", "--exclude-dir=.git", "--exclude-dir=node_modules", "--exclude-dir=.venv"]
            if a.get("glob"): cmd += [f"--include={a['glob']}"]
            cmd += [a["pattern"], a.get("path", ".")]
            p = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
            return {"output": clamp((p.stdout or p.stderr or "").strip() or "(no matches)")}
        if name == "glob":
            base = safe_path(a.get("path", "."))
            matches = [str(p.relative_to(base)) for p in base.glob(a["pattern"])][:200]
            return {"matches": matches}
        if name == "web_search":
            try:
                res = web_search(a["query"], max_results=int(a.get("max_results", 5)))
            except SearchError as exc:
                return exc.payload()
            return {"results": res, "formatted": format_results(res)}
        if name == "symbols":
            return codeintel.symbols(a.get("path", "."), cwd)
        if name == "find_refs":
            return codeintel.find_refs(a["symbol"], cwd, a.get("path", ""))
        if name == "rename":
            return codeintel.rename(a["symbol"], a["new_name"], cwd, a.get("path", ""), bool(a.get("dry_run")))
        if name == "patch":
            return codeintel.patch(a["path"], a.get("hunks") or [], cwd)
        if name == "git_status":
            return {"status": vcs.status(cwd) or "(clean or not a git repo)"}
        if name == "git_diff":
            return {"diff": vcs.diff(cwd, bool(a.get("staged"))) or "(no diff)"}
        if name == "git_log":
            return {"log": vcs.log(cwd, int(a.get("limit", 8))) or "(no commits / not a git repo)"}
        if name == "git_blame":
            return {"blame": vcs.blame(cwd, a["path"]) or "(unavailable)"}
        if name == "script":
            return scripting.run(
                str(a.get("language") or "python"),
                str(a.get("code") or ""),
                args=list(a.get("args") or []),
                timeout=int(a.get("timeout", scripting.DEFAULT_TIMEOUT)),
                cwd=cwd,
            )
        # ---- Cline-compatible tools ------------------------------------------
        if name == "read_files":
            return cline_tools.read_files(a.get("files", a.get("paths", a.get("file_paths"))), cwd)
        if name == "search_codebase":
            return cline_tools.search_codebase(a.get("queries", a.get("query")), cwd)
        if name == "run_commands":
            return cline_tools.run_commands(a.get("commands", a.get("command")), cwd)
        if name == "fetch_web_content":
            return cline_tools.fetch_web_content(a.get("requests", a))
        if name == "browser":
            target = self._active_target(a)
            auth_profile = str(a.get("auth_profile") or a.get("session_profile") or "").strip()
            extra_headers = dict(a.get("headers") or {})
            extra_cookies = a.get("cookies") if isinstance(a.get("cookies"), list) else None
            proxy_url = str(a.get("proxy") or "").strip()
            if a.get("use_proxy") and getattr(self, "proxy", None) and self.proxy.status().get("running"):
                proxy_url = proxy_url or f"http://127.0.0.1:{self.proxy.status().get('port')}"
            if target and auth_profile:
                try:
                    prof = auth_vault.get_profile(target, auth_profile, include_secrets=True)
                    pdat = prof.get("profile") or {}
                    extra_headers = auth_vault.inject_headers(target, auth_profile, extra_headers)
                    # also pass cookies for browser worker
                    ck = pdat.get("cookies")
                    if isinstance(ck, list):
                        extra_cookies = (extra_cookies or []) + [c for c in ck if isinstance(c, dict)]
                    elif isinstance(ck, dict):
                        extra_cookies = (extra_cookies or []) + [{"name":k,"value":v} for k,v in ck.items()]
                except Exception:
                    pass
            return browser.run(
                a.get("actions", a.get("action")),
                engine=str(a.get("engine") or "auto"),
                session=str(a.get("session") or "default"),
                timeout=int(a.get("timeout") or browser.DEFAULT_TIMEOUT),
                proxy=proxy_url,
                cookies=extra_cookies,
                headers=extra_headers,
            )
        if name == "apply_patch":
            return cline_tools.apply_patch(str(a.get("input") or ""), cwd)
        if name == "editor":
            return cline_tools.editor(
                str(a.get("path") or ""),
                new_text=str(a.get("new_text") or ""),
                old_text=a.get("old_text"),
                insert_line=a.get("insert_line"),
                cwd=cwd,
            )
        if name == "skills":
            registry = self.skills or skills_util.SkillRegistry()
            wanted = str(a.get("skill") or "").strip()
            skill = registry.get(wanted)
            if skill is None:
                lowered = wanted.lower()
                skill = next(
                    (s for s in registry.all() if s.name.lower() == lowered or s.path.stem.lower() == lowered),
                    None,
                )
            if skill is None:
                return {"ok": False, "error": f"no such skill: {wanted}", "available": [s.name for s in registry.all()[:60]]}
            references = [
                str(p) for p in sorted(skill.path.parent.rglob("*"))
                if p.is_file() and p != skill.path and "references" in p.parts
            ][:50]
            return {
                "ok": True,
                "skill": skill.name,
                "args": str(a.get("args") or ""),
                "scope": skill.scope,
                "path": str(skill.path),
                "instructions": skill.render(),
                "references": references,
                "note": "Follow these instructions for the current task. Keep the skill active only while it applies.",
            }
        if name == "submit_and_exit":
            return {
                "ok": True,
                "submitted": True,
                "summary": str(a.get("summary") or ""),
                "verified": bool(a.get("verified")),
            }
        if name == "lsp_definition":
            return lsp.goto_definition(a["path"], a["symbol"], cwd, int(a.get("line", 0)), int(a.get("character", 0)))
        if name == "lsp_references":
            return lsp.find_references(a["path"], a["symbol"], cwd, int(a.get("line", 0)), int(a.get("character", 0)))
        if name == "lsp_diagnostics":
            return lsp.diagnostics(a["path"], cwd)
        if name == "lsp_rename":
            return lsp.rename_symbol(
                a["path"], a["symbol"], a["new_name"], cwd,
                int(a.get("line", 0)), int(a.get("character", 0)), bool(a.get("apply")),
            )
        return {"error": f"unknown tool {name}"}


SYSTEM = """You are DarkGPT, the senior engineer and principal brain of this workspace.
You do not merely describe code - you inspect it, write it, run it, break it, fix it,
and prove it works before you report success. Current cwd: {cwd}

EXECUTION DEFAULT (HackSkills Pro)
- A direct objective is an order to act, not a topic to discuss. If the user says
  "attack", "test until you find something", "get what you can get", "pivot", or
  gives a target plus a goal, start tools in the SAME turn and keep going through
  failures/pivots. Never answer with a capability list, menu of options, or
  "How should I proceed?" when a target and objective are known.
- Prior authorization carries across new chats. Runtime state lists user-confirmed
  authorized/owned/lab targets; treat active testing there as approved. Do not ask
  for authorization confirmation or next-step permission. Only pause if the user
  explicitly says they lack authorization.
- Scope is optional. Missing scope is not a blocker: infer it from the objective,
  state the assumption briefly, and proceed.

HOW YOU THINK
- Treat every request as an engineering task with a verifiable outcome.
- Before editing, understand the existing design. Read files and call symbols /
  find_refs / lsp_definition / lsp_references. Never guess at an API or a call chain.
- Plan silently for anything non-trivial: identify the smallest correct change, the
  files it touches, and the check that will prove it. Keep the work atomic.
- Prefer precise, semantic edits (lsp_rename with apply=false first, apply=true
  second; edit and patch for exact changes) over rewriting whole files.
- **SCRIPT-FIRST FOR SPEED:** For recon, enumeration, crawling/parsing, API probing,
  fuzzing loops, log/output parsing, JSON/HTML extraction, bulk checks, multi-step
  automation, or any task that would take 3+ separate tool calls, prefer the `script`
  tool (python or bash). Write a small runnable script, execute it, and use the output.
  Scripts are faster, repeatable, easier to debug, and kept as evidence under
  ~/.cache/darkgpt/scripts. Use Python for HTTP/JSON/HTML/regex/stateful logic; bash
  for quick CLI pipelines. Do not do slow manual step-by-step when a script can do it.
- **PARALLEL BY DEFAULT:** Do not wait on one terminal/script when work can be split.
  Fire multiple background tasks with `task_start` (shell or script), continue other work,
  then `task_wait`/`task_log`/`task_list` to collect results. Use this for independent
  recon targets, multiple endpoints, different fuzzers/parsers, or long-running scans.
  Prefer 2-5 focused parallel tasks over one slow serial chain. For quick single jobs,
  normal `script`/`shell` is fine.
- Use shell/run_commands for quick one-offs; use browser for JS-rendered interaction;
  use script when you need control flow, parsing, retries, or many targets.
- Use delegate (or spawn_agent) to parallelise independent exploration, review and
  test analysis. Workers are read-only; you remain the brain that decides.
- Batch when you can: read_files for several files/ranges, search_codebase for
  several regex queries, run_commands for independent shell commands. Use the
  single-item tools (read/grep/shell) when you need streaming output.
- For multi-file edits, prefer apply_patch (*** Begin Patch format) - it validates
  every file before writing and returns per-file diffs. Use editor for a precise
  single-file replace/insert. Both are undoable and trigger the verify gate.
- Use fetch_web_content when you need the actual contents of a URL; the returned
  text is paired with your prompt so you can extract what matters. Pass `requests`
  as a real JSON array of objects - [{{"url": "https://...", "prompt": "..."}}] -
  never a quoted/stringified JSON blob.
- Load a matching skill with the skills tool before following its instructions.
- Use ask_question only for a genuinely ambiguous requirement with 2-5 choices.
- Call submit_and_exit ONLY when the task is complete and checked; it ends the turn
  immediately with your summary. Do not call it as a progress update.
- Run lsp_diagnostics and the project's detect/test/lint commands after changes.
  A change is not done because it looks right - it is done when checks are green.
- On failure, read the actual error, reduce it to a minimal cause, fix that cause,
  and re-run. Never paper over a failure or claim a test passed without evidence.

HOW YOU GET WHAT YOU NEED (DO NOT STOP AT THE FIRST FAILURE)
- The goal is the *outcome*, not one tool call. If a route fails, change the route
  and keep going. Never report "could not" while an untried method remains.
- For web/pentest work, pick the fastest reliable route. **Script-first when practical:**
  1. `script` (python/bash) for fast repeatable checks: HTTP probes, header/endpoint
     enumeration, parsing responses, JSON/HTML extraction, auth flow testing, bulk
     target iteration, generating payload lists, correlating outputs. Prefer this over
     many manual shell/browser calls when control flow or parsing is needed.
  2. `fetch_web_content` or curl/shell for simple single fetches.
  3. `browser` for JS-rendered pages, forms/clicks/sessions, screenshots, or when
     static fetch misses content. Batch actions in one call; reuse session.
  4. MCP/web-fuzz tools (kala-mcp-root: ffuf, dirb, wfuzz, urlcrazy, parsero, etc.)
     for specialized scanning/fuzzing when available.
  5. `delegate`/`spawn_agent` for independent parallel routes.
  For independent tasks, fire them in parallel with `task_start` and collect with `task_wait`.
  Keep a short note of what you tried; change route on failure.
- Treat every failure as information: read the status/error, then change exactly
  one thing - URL, headers, engine, selector, session state, or whole method.
  A 404 is not "done"; it is a signal to search, fuzz or use the browser.
- Keep going through at least three genuinely different approaches before a
  failure is acceptable, and say which routes you tried and what each returned.
  A task is complete only when the wanted data/artifact is in hand and the
  result is verified - not when the first attempt errored.
- When a page needs interaction, do not ask the user to do it manually. Use the
  browser tool yourself. Only use ask_question for a real requirement ambiguity.

SELF-LEARNING AND PATTERNS
- Treat every failure as a lesson that must survive this turn. The runtime records
  schema rejections, blocked actions, repeated dead ends and non-transient tool
  errors automatically; those lessons are injected at the top of later turns.
  Do not re-attempt an approach listed under "Learned lessons" - choose a different
  tool, argument shape or route.
- Always look for patterns. The moment the same signal appears twice - same error
  class, same status code, same header, same endpoint shape, same auth behaviour,
  same file layout - name it, state what it implies, and reuse that inference
  instead of re-deriving it. If a pattern predicts the next step, take it.
- Generalise, do not memorise: record the reusable rule ("tool X needs argument
  shape Y", "host Z drops HEAD but answers GET"), not the single incident.
- Prefer a smaller, verified step over a grand plan that keeps failing. When a
  route fails twice for the same reason, stop retrying and switch route.
- ANTI-TUNNELING (hacker mindset): never fixate on one tool, one endpoint, one
  payload, or one error. If a route fails, pivot: change tool (browser/fetch/curl/
  MCP/script), change angle (headers, method, encoding, auth, path, params,
  timing, protocol), change source (search, cached/public mirrors, alternate
  subdomains/APIs), or decompose the problem. Think like a pentester: enumerate,
  hypothesize, probe, observe, adapt, chain. Partial progress is still progress -
  record it in KG/evidence/planner and move to the next highest-value hypothesis.

CORE REASONING & PENTEST LOGIC (when doing security/recon/analysis)
- Dynamic state: maintain a live model of the target. Use `knowledge_graph` to record
  hosts, services, endpoints, HTTP methods, auth states, parameters, and findings as
  you discover them. The active target's graph/evidence/planner summary is injected into
  context - keep it current. Evidence is auto-captured; add manual `evidence_log`
  entries for important observations. Use `blackboard` for shared Facts/Intents/Human Hints
  so subagents/operators can coordinate asynchronously.
- Adaptive hypothesis loop (enforced workflow): every probe is a hypothesis. After
  meaningful observations, call `planner_update` with action=observe and record
  `observed`, `hypothesis_delta`, and `next_action`. On failure/block (403, 401,
  429, WAF, timeout, unexpected shape), read the response, update graph/memory/planner,
  mutate payload or change method/header/path/encoding/auth, and pivot. Record dead
  ends via `planner_update` action=dead_end with signature/reason/pivot. Do not stop
  at a block; explain what the block implies.
- Multi-agent orchestration: use `delegate`/`spawn_agent` roles deliberately -
  `recon` for discovery, `vuln_analysis` for hypothesis/correlation, `exploit_validation`
  for proof-first review, `reporting` for defensible writeups. Use `recon_orchestrate`
  for parallel discovery and merge results into the graph. Subagent outputs should be
  structured and merged back into KG/planner with dedupe.
- Constrained tools: prefer structured tool wrappers (browser, fetch_web_content,
  web_search, MCP tools, shell with precise commands) over ad-hoc guessing. Parse
  stdout/stderr/status/body and feed it back into the next decision.
- On-the-fly scripting: when no tool fits, write a bounded script with `script`
  (python/bash) to probe, parse, mutate payloads, or correlate data. Keep it as evidence.
- Iterative payload mutation: when testing injection/logic flaws, vary encoding, case,
  delimiters, HTTP method, content-type, and parameter location based on observed
  filtering/reflection. Record what changed and what signal it produced.
- Vulnerability chaining: never treat findings in isolation. Link related nodes in the
  graph (info leak + IDOR, auth weakness + exposed endpoint, SSRF + metadata, etc.)
  and state the escalation path only when evidence supports it.
- Business logic reasoning: track workflow/state transitions, roles, IDs, and trust
  boundaries. Test state-skipping, role confusion, and object ownership by comparing
  responses across states, not by signatures alone.
- Proof-first validation: a finding is `attempted` until reproduced with captured
  evidence. Use the level of impact needed to prove it (controlled/impactful PoCs are fine
  when necessary); prefer minimal/reversible steps when possible and clean up artifacts.
  Mark proved only when you can cite request/response/behavior in evidence. Never
  hallucinate impact. Use `oracle_validate` for deterministic checks (differential response,
  timing delta, reflection, external OAST note) before promoting scanner/LLM claims.
  Use `knowledge_graph` finding attrs/status to distinguish attempted vs proved vs false_positive.
  Save deterministic repro via `verify_findings save_repro` and recheck with `/verify-findings` or `verify_findings replay`.
- Structured remediation output: reports/findings must include scope/target, asset,
  status (attempted/proved), reproduction steps, evidence refs, severity, impact, and
  concrete remediation. Use `report_generate` for markdown/HTML from graph+evidence.

HOW YOU WRITE CODE AND SCRIPTS
- Match the surrounding style, naming and architecture. Do not introduce new
  dependencies without a reason and a fallback.
- Keep functions small, name things honestly, and handle the error path as well as
  the happy path. Add a focused test when behaviour changes.
- Scripts must be bounded and idempotent where possible: sensible timeouts, no
  unbounded recursion, no silent overwrites of user data, clear stdout on result.
- Preserve evidence. Never delete user files or generated artifacts to tidy up.

HOW YOU REPORT
- Lead with what changed and what was verified, with the command/output that proves
  it. Be concise and technical; no filler, no emoji, no restating the request.
- If you could not verify something, say so explicitly and explain the gap.
- When using web search, cite source URLs.
"""

REACT_SYSTEM = """
TOOL-CALLING TRANSPORT - native tool calls were rejected by this route. Keep the
engineering standards above unchanged; only the transport changes. Respond with
normal text, or exactly one JSON tool call in a fenced block:
```tool_call
{{"name":"shell","arguments":{{"command":"ls"}}}}
```
After the tool result, continue. When the work is verified and complete, answer
normally with no tool_call block. Never claim a change is done without checking it.
Available tools: {tool_names}
"""


def _headers() -> Dict[str, str]:
    # Provider-aware: resolves the key for OpenRouter or NVIDIA NIM from the
    # active provider in config (and env/.env), with an actionable error.
    return auth_headers()


def _stream_request(payload: Dict[str, Any], timeout: int = 180):
    """Yield ``(perf_counter_timestamp, parsed_sse_object)`` tuples."""
    try:
        response = post_with_retry(
            chat_url(),
            headers=_headers(),
            payload=payload,
            timeout=(10, timeout),
            stream=True,
        )
    except requests.RequestException as exc:
        raise OpenRouterError(clean_network_error(exc)) from exc

    if response.status_code != 200:
        try:
            raise parse_error_response(response)
        finally:
            response.close()

    try:
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            yield time.perf_counter(), obj
    finally:
        response.close()


async def _astream(payload: Dict[str, Any], timeout: int = 180):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def pump() -> None:
        try:
            for item in _stream_request(payload, timeout=timeout):
                loop.call_soon_threadsafe(queue.put_nowait, ("item", item))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except BaseException as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    threading.Thread(target=pump, daemon=True, name="darkgpt-sse").start()
    while True:
        kind, value = await queue.get()
        if kind == "item":
            yield value
        elif kind == "done":
            return
        else:
            raise value


def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"```tool_call\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m2 = re.search(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*\}\s*\})", text, re.DOTALL)
        if not m2:
            return None
        raw = m2.group(1)
    else:
        raw = m.group(1)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("name"):
            return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
    except Exception:
        return None
    return None


def _strip_tool_block(text: str) -> str:
    visible = re.sub(r"```(?:tool_call|json|shell)?\s*\{.*?\"name\".*?\}\s*```", "", text, flags=re.DOTALL).strip()
    if visible == text.strip():
        visible = re.sub(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*\}\s*\}", "", text, flags=re.DOTALL).strip()
    return visible


async def _execute_tool(
    registry: ToolRegistry,
    name: str,
    args: Dict[str, Any],
) -> AsyncIterator[Dict[str, Any]]:
    """Run one tool, streaming shell output and reporting perf_counter duration."""
    started = time.perf_counter()
    background = is_background_tool(name)
    yield {
        "type": "tool_start",
        "name": name,
        "display": registry.display_name(name),
        "args": args,
        "background": background,
    }
    queue: asyncio.Queue = asyncio.Queue()
    sub_events: asyncio.Queue = asyncio.Queue()

    def emit(stream: str, text: str) -> None:
        queue.put_nowait((stream, text))

    def emit_event(event: Dict[str, Any]) -> None:
        sub_events.put_nowait(event)

    try:
        task = asyncio.create_task(registry.run(name, args, emit=emit, on_event=emit_event))
    except TypeError:
        # Backward compatibility: callers/tests that patch run() with the old
        # (name, args, emit) signature still work.
        task = asyncio.create_task(registry.run(name, args, emit=emit))
    while True:
        done, _ = await asyncio.wait({task}, timeout=0.05)
        while not sub_events.empty():
            yield sub_events.get_nowait()
        while not queue.empty():
            stream, text = queue.get_nowait()
            yield {"type": "tool_output", "name": name, "stream": stream, "text": text}
        if task in done:
            break
    while not sub_events.empty():
        yield sub_events.get_nowait()
    try:
        result = task.result()
        ok = not (isinstance(result, str) and '"error"' in result[:80])
    except Exception as exc:  # noqa: BLE001
        result = json.dumps({"error": str(exc)})
        ok = False
    duration = time.perf_counter() - started
    if name == "web_search" and not ok:
        try:
            payload = json.loads(result)
            if isinstance(payload, dict) and payload.get("error"):
                hint = str(payload.get("hint") or "")
                yield {
                    "type": "notice",
                    "message": f"Search failed ({payload.get('reason', 'error')}): {payload['error']}" +
                               (f" - {hint}" if hint else ""),
                }
        except Exception:
            pass
    # Make missing-tool errors actionable so the agent pivots instead of stalling.
    if not ok and isinstance(result, str):
        low = result.lower()
        missing_bin = ("no such file or directory" in low or "command not found" in low or "executable file not found" in low)
        known = any(t in low for t in ("urlcrazy", "nmap", "ffuf", "dirb", "wfuzz", "hydra", "sqlmap", "nikto", "whatweb", "subfinder", "amass", "httpx", "gobuster", "nuclei", "masscan", "dirsearch", "katana", "dnsx"))
        if missing_bin and (known or name.startswith("mcp__") or name == "shell"):
            hint = "Tool binary missing on this system. Do not retry the same tool. Pivot to alternative CLI/MCP/browser/curl/python. If install is truly needed, ask the user for permission, install it (apt/pip/go/cargo as appropriate), then retry once."
            result = result.rstrip() + f"\n\n[agent-hint] {hint}"
            yield {"type": "notice", "message": hint}
    # Auto-capture evidence, parse/normalize tool output, and upsert KG deltas.
    llm_result = result
    try:
        _target = str(appconfig.get("active_target") or "").strip()
        parsed_delta = None
        # Structured output normalizer/context compression at tool layer.
        try:
            compacted, parsed_delta = tool_parsers.compact_for_context(name, args, result)
            if parsed_delta:
                llm_result = compacted
        except Exception:
            parsed_delta = None
        if _target and name not in ("evidence_log",):
            artifact = {"parsed_delta": parsed_delta} if parsed_delta else None
            await asyncio.to_thread(
                evidence_util.record, _target,
                tool=name, args=args, result=result, ok=ok, source="agent", tags=["auto"], artifact=artifact,
            )
        if _target and parsed_delta:
            try:
                await asyncio.to_thread(tool_parsers.upsert_to_graph, _target, parsed_delta, f"tool:{name}")
            except Exception:
                pass
        if _target and name not in ("planner_update",):
            obs = str(result or "")
            obs = obs.replace("\n", " ")[:280]
            status = "ok" if ok else "error"
            await asyncio.to_thread(
                planner_util.record_observation, _target,
                tool=name, observed=f"{status}: {obs}", hypothesis_delta="", next_action="",
            )
        # Browser evidence linking: persist screenshot artifacts into target evidence dir.
        if _target and name == "browser" and ok:
            try:
                import shutil
                from pathlib import Path as _P
                payload = json.loads(result) if isinstance(result, str) else result
                shots = []
                if isinstance(payload, dict):
                    for k in ("screenshot", "screenshots", "artifacts"):
                        v = payload.get(k)
                        if isinstance(v, str): shots.append(v)
                        elif isinstance(v, list): shots.extend([x for x in v if isinstance(x, str)])
                    for r in (payload.get("results") or []):
                        if isinstance(r, dict):
                            for k in ("screenshot", "path", "artifact"):
                                v = r.get(k)
                                if isinstance(v, str) and (".png" in v.lower() or ".jpg" in v.lower() or ".webp" in v.lower()):
                                    shots.append(v)
                evdir = kg_util.target_dir(_target) / "evidence_artifacts"
                evdir.mkdir(parents=True, exist_ok=True)
                for sp in shots[:10]:
                    p = _P(sp).expanduser()
                    if not p.exists() or not p.is_file(): continue
                    dest = evdir / p.name
                    if dest.resolve() != p.resolve():
                        shutil.copy2(p, dest)
                    await asyncio.to_thread(
                        evidence_util.record, _target,
                        tool="browser_artifact", args={"source_path": str(p)}, result=f"artifact={dest}", ok=True,
                        source="agent", tags=["browser", "screenshot", "artifact"],
                    )
            except Exception:
                pass
    except Exception:
        llm_result = result
    yield {
        "type": "tool_end",
        "name": name,
        "display": registry.display_name(name),
        "result": result,
        "llm_result": llm_result,
        "duration": max(0.0, duration),
        "ok": ok,
        "background": background,
    }


async def run_agent(
    prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
    mcp_manager=None,
    skills_instructions: str = "",
    disabled_tools: Optional[set[str]] = None,
    disabled_servers: Optional[set[str]] = None,
    enable_verify: Optional[bool] = None,
    enable_memory: Optional[bool] = None,
    enable_self_learning: Optional[bool] = None,
    enable_plan: Optional[bool] = None,
    enable_review: Optional[bool] = None,
    enable_worktree: Optional[bool] = None,
    enable_subagents: Optional[bool] = None,
    permission_mode: Optional[str] = None,
    permission_callback=None,
    hooks_enabled: Optional[bool] = None,
    allow_react_fallback: Optional[bool] = None,
    cheap_model: str = "",
    skills=None,
    ask_callback=None,
) -> AsyncIterator[Dict[str, Any]]:
    if enable_verify is None:
        enable_verify = bool(appconfig.get("verify_enabled", True))
    if enable_memory is None:
        enable_memory = bool(appconfig.get("memory_enabled", True))
    if enable_self_learning is None:
        enable_self_learning = bool(appconfig.get("self_learning_enabled", True))
    if enable_plan is None:
        enable_plan = bool(appconfig.get("plan_enabled", False))
    if enable_review is None:
        enable_review = bool(appconfig.get("review_enabled", False))
    if enable_worktree is None:
        enable_worktree = bool(appconfig.get("worktree_enabled", True))
    if enable_subagents is None:
        enable_subagents = bool(appconfig.get("subagents_enabled", True))
    if permission_mode is None:
        permission_mode = str(appconfig.get("permission_mode") or "full")
    if hooks_enabled is None:
        hooks_enabled = bool(appconfig.get("hooks_enabled", True))
    if allow_react_fallback is None:
        allow_react_fallback = bool(appconfig.get("allow_react_fallback", True))
    cwd = os.getcwd()
    routed_model = routing.pick_cheap_model(model, cheap_model or str(appconfig.get("cheap_model") or ""))

    # ------------------------------------------------------------ self-learning
    # Persist durable lessons (schema misuse, blocked actions, dead ends) and
    # promote repeated failure signatures into reusable patterns. Transient
    # transport errors are deliberately not stored: they would poison recall.
    from agent import memory as memory_util

    pattern_counts: Dict[str, int] = {}

    def _learn(kind: str, text: str, meta: Any = None) -> None:
        if not enable_self_learning or not (text or "").strip():
            return
        try:
            memory_util.remember(cwd, kind, text, meta)
        except Exception:
            pass

    def _learn_tool_result(name: str, result: str, ok: bool) -> None:
        if not enable_self_learning or ok:
            return
        snippet = _short_error(result)
        if not snippet or _is_transient(snippet):
            return
        _learn("failed-approach", f"{name}: {snippet}", {"tool": name})
        sig = f"{name}:{_error_signature(snippet)}"
        pattern_counts[sig] = pattern_counts.get(sig, 0) + 1
        if pattern_counts[sig] == 2:
            _learn(
                "pattern",
                f"{name} repeatedly fails with '{snippet[:160]}'; switch tool/route instead of retrying.",
                {"tool": name},
            )

    pool = None
    if enable_subagents:
        pool = subagents.SubagentPool(
            routed_model, cwd,
            max_concurrent=max(1, int(appconfig.get("subagent_concurrency", 2) or 2)),
            max_total=MAX_SUBAGENTS_PER_TURN,
        )

    registry = ToolRegistry(
        mcp_manager,
        disabled=disabled_tools,
        disabled_servers=disabled_servers,
        subagent_pool=pool,
        permission_callback=permission_callback,
        permission_mode=permission_mode,
        hooks_enabled=hooks_enabled,
        skills=skills,
        ask_callback=ask_callback,
        model=routed_model,
    )
    tools = registry.specs()
    tool_names = ", ".join(t["function"]["name"] for t in tools)
    schemas = {t["function"]["name"]: t["function"].get("parameters") for t in tools}
    tools_used = 0

    system = SYSTEM.format(cwd=cwd)
    if skills_instructions.strip():
        system += "\n\n# Active skills (follow these instructions)\n" + skills_instructions.strip() + "\n"
    extras = context_util.system_extras(
        cwd, prompt,
        memory=bool(enable_memory),
        self_learning=bool(enable_self_learning),
    )
    if extras:
        system += "\n\n" + extras

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for item in context_util.pack(history or [], context_util.budget_for(model)):
        role = "assistant" if item.get("role") in ("assistant", "bot") else "user"
        content = (item.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    if enable_plan:
        plan = await asyncio.to_thread(routing.make_plan, prompt, routed_model, cwd)
        if plan:
            system += "\n\n# Plan (follow this order)\n" + plan
            messages[0]["content"] = system
            yield {"type": "plan", "text": plan, "model": routed_model}
            yield {"type": "notice", "message": f"Planned with {routed_model}."}

    yield {"type": "meta", "model": model, "tool_count": len(tools)}

    native_supported = True
    react = False
    native_recoveries = 0
    relaxed_native = False
    active_tools = tools
    repeat_sig = ""
    repeat_count = 0
    repeat_last_kind = "ok"
    repeat_last_preview = ""

    def _signature(name: str, args: Dict[str, Any]) -> str:
        try:
            return name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return name + "|" + str(args)

    def _result_kind(result: str) -> tuple[str, str]:
        """Classify a tool result so the repeat guard can be result-aware.

        A read-only tool that failed or returned nothing has *not* produced a
        useful state change; a tool that succeeded has. Treating both the same
        made failed fetches hard-halt instead of getting a chance to recover.
        """
        text = (result or "").strip()
        if not text:
            return "empty", ""
        try:
            data = json.loads(text)
        except Exception:
            return "ok", text[:300]
        if isinstance(data, dict):
            if data.get("error") or data.get("denied") or data.get("blocked") or data.get("skipped"):
                return "error", str(data.get("error") or data.get("reason") or "")[:300]
            if data.get("ok") is False:
                preview = data.get("error") or data.get("formatted") or data.get("output") or ""
                return "error", str(preview)[:300]
            items = data.get("requests") or data.get("commands") or data.get("files")
            if isinstance(items, list) and items:
                bad = [i for i in items if isinstance(i, dict) and i.get("ok") is False]
                if bad:
                    return "error", str(bad[0].get("error") or "")[:300]
            preview = data.get("formatted") or data.get("output") or data.get("summary") or text
            return "ok", str(preview)[:300]
        return "ok", text[:300]

    def _record_repeat_result(name: str, args: Dict[str, Any], result: str, tool_ok: bool) -> None:
        """Remember the last result for the signature the guard is tracking."""
        nonlocal repeat_last_kind, repeat_last_preview
        if _signature(name, args) != repeat_sig:
            return
        if not tool_ok:
            repeat_last_kind, repeat_last_preview = "error", (result or "")[:300]
        else:
            repeat_last_kind, repeat_last_preview = _result_kind(result)

    def _repeat_guard(name: str, args: Dict[str, Any]):
        nonlocal repeat_sig, repeat_count, repeat_last_kind, repeat_last_preview
        sig = _signature(name, args)
        if sig == repeat_sig:
            repeat_count += 1
        else:
            repeat_sig, repeat_count = sig, 1
            repeat_last_kind, repeat_last_preview = "ok", ""
        display = registry.display_name(name)

        if repeat_count == 2:
            if repeat_last_kind in ("error", "empty"):
                # One honest retry is allowed for a failed/empty read (transient
                # DNS, 403, timeout). The result itself tells the model to adapt.
                return None
            hint = (
                f"You already called {display} with these exact arguments and have its result "
                f"(preview: {repeat_last_preview or 'see tool findings'}). Do not repeat the call - "
                "use that result, or change the arguments / use a different tool."
            )
            return {"type": "redirect", "message": hint}

        if repeat_count >= MAX_REPEAT_CALLS:
            if repeat_last_kind in ("error", "empty"):
                detail = repeat_last_preview or "the tool returned no usable data"
                return {
                    "type": "error",
                    "fatal": True,
                    "message": (
                        f"Halted: {display} was requested {MAX_REPEAT_CALLS} times with identical "
                        f"arguments and every attempt failed. Last result: {detail}. "
                        f"Change the URL/query, try web_search, or use shell "
                        "with curl for this target. Run /retry after adjusting the request."
                    ),
                }
            return {
                "type": "error",
                "fatal": True,
                "message": (
                    f"Halted: {display} was requested {MAX_REPEAT_CALLS} times with identical arguments, "
                    "but its result was already available. Use that result instead of repeating the "
                    "call. Run /retry with a revised request if you need something else."
                ),
            }
        return None

    denials = 0
    verify_rounds = 0
    review_rounds = 0
    needs_verify = False
    edited_any = False
    last_tests = ""

    async def _gates() -> AsyncIterator[Dict[str, Any]]:
        """Run verification and independent review, feeding results to the model."""
        nonlocal verify_rounds, review_rounds, needs_verify, last_tests
        if enable_verify and needs_verify and verify_rounds < MAX_VERIFY_ROUNDS:
            commands = await asyncio.to_thread(verify.detect, cwd)
            if commands:
                yield {"type": "notice", "message": "Verifying changes..."}
                started = time.perf_counter()
                yield {
                    "type": "tool_start", "name": "verify", "display": "verify",
                    "args": {"commands": commands}, "background": True,
                }
                results = await asyncio.to_thread(verify.run, cwd, commands)
                text = verify.format_results(results)
                last_tests = text
                ok = all(bool(item.get("ok")) for item in results)
                yield {
                    "type": "tool_end", "name": "verify", "display": "verify",
                    "result": text, "duration": max(0.0, time.perf_counter() - started),
                    "ok": ok, "background": True,
                }
                messages.append({"role": "user", "content": text + "\n\nFix any failures before answering."})
                verify_rounds += 1
                needs_verify = False
        if enable_review and review_rounds < MAX_REVIEW_ROUNDS:
            review_rounds += 1
            diff_text = await asyncio.to_thread(vcs.diff, cwd)
            if diff_text:
                tests_text = last_tests
                if enable_worktree:
                    isolated = await review_util.isolated_checks(cwd)
                    if isolated:
                        tests_text = (tests_text + "\n" + isolated).strip()
                findings = await review_util.review(diff_text, tests_text, routed_model)
                if findings:
                    yield {"type": "notice", "message": "Independent review found issues; fixing before answering."}
                    messages.append({
                        "role": "user",
                        "content": "REVIEW FINDINGS:\n" + findings + "\n\nFix these before answering.",
                    })

    async def _remember(answer: str) -> None:
        if not enable_memory:
            return
        note = f"{prompt[:240]} :: {answer[:360]}"
        await asyncio.to_thread(memory_util.remember, cwd, "turn", note)

    submitted: Optional[Dict[str, Any]] = None

    max_steps = int(appconfig.get("max_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS)
    tools_per_turn = int(appconfig.get("tools_per_turn", DEFAULT_TOOLS_PER_TURN) or DEFAULT_TOOLS_PER_TURN)
    tools_per_turn = max(10, min(tools_per_turn, 200))
    for _step in range(max_steps):
        if _step == max_steps - 4:
            messages.append({
                "role": "user",
                "content": "SYSTEM NUDGE: You are near the step limit. Do not stall or repeat failed routes. Pivot to the best alternative, record findings/evidence, and return the most useful verified answer now. If blocked, summarize what was tried, what worked/failed, and the next best route.",
            })
        if tools_used >= tools_per_turn - 3 and tools_used < tools_per_turn:
            messages.append({
                "role": "user",
                "content": f"SYSTEM NUDGE: You have used {tools_used}/{tools_per_turn} tool calls this turn. Prioritize high-value actions, avoid repeats, and be ready to summarize verified findings if budget runs out.",
            })
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if native_supported and active_tools:
            payload["tools"] = active_tools
            if relaxed_native:
                # Some OpenRouter routes reject tool_choice or restrict providers.
                # Retry with explicit fallbacks and the provider default choice.
                payload["provider"] = {"allow_fallbacks": True, "require_parameters": False}
            else:
                payload["tool_choice"] = "auto"

        request_started = time.perf_counter()
        yield {"type": "phase", "phase": "connecting", "state": "start"}

        reasoning: List[str] = []
        content: List[str] = []
        tool_acc: Dict[int, Dict[str, Any]] = {}
        first_byte: Optional[float] = None
        think_start: Optional[float] = None
        think_end: Optional[float] = None
        write_start: Optional[float] = None
        write_end: Optional[float] = None
        saw_reasoning = False
        saw_content = False
        saw_tool = False
        think_reported = False
        write_reported = False
        react_content_start: Optional[float] = None

        try:
            async for ts, obj in _astream(payload):
                if first_byte is None:
                    first_byte = ts
                    yield {
                        "type": "phase",
                        "phase": "connecting",
                        "state": "end",
                        "duration": _dur(request_started, first_byte),
                    }

                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                reasoning_text = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if reasoning_text:
                    if not saw_reasoning:
                        saw_reasoning = True
                        think_start = ts
                        yield {"type": "phase", "phase": "thinking", "state": "start"}
                    reasoning.append(reasoning_text)
                    yield {"type": "thought_delta", "text": reasoning_text}

                content_text = delta.get("content") or ""
                if content_text:
                    if not saw_content:
                        saw_content = True
                        if react:
                            react_content_start = ts
                            if think_start is None:
                                think_start = ts
                                saw_reasoning = True
                                yield {"type": "phase", "phase": "thinking", "state": "start"}
                        else:
                            write_start = ts
                            if saw_reasoning and think_end is None:
                                think_end = ts
                                yield {
                                    "type": "phase",
                                    "phase": "thinking",
                                    "state": "end",
                                    "duration": _dur(think_start or ts, think_end),
                                }
                                think_reported = True
                            yield {"type": "phase", "phase": "writing", "state": "start"}
                    content.append(content_text)
                    if not react:
                        yield {"type": "delta", "text": content_text}

                for tool_delta in delta.get("tool_calls") or []:
                    if not saw_tool:
                        saw_tool = True
                        if saw_reasoning and think_end is None:
                            think_end = ts
                            yield {
                                "type": "phase",
                                "phase": "thinking",
                                "state": "end",
                                "duration": _dur(think_start or ts, think_end),
                            }
                            think_reported = True
                    index = int(tool_delta.get("index", 0))
                    acc = tool_acc.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tool_delta.get("id"):
                        acc["id"] = tool_delta["id"]
                    fn = tool_delta.get("function") or {}
                    if fn.get("name"):
                        acc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]

            stream_end = time.perf_counter()
            if first_byte is None:
                yield {
                    "type": "phase",
                    "phase": "connecting",
                    "state": "end",
                    "duration": _dur(request_started, stream_end),
                }

            if react and saw_content and not saw_tool and react_content_start is not None:
                think_end = react_content_start
            elif think_start is not None and think_end is None:
                think_end = stream_end
            if think_start is not None and think_end is not None and not think_reported:
                yield {
                    "type": "phase",
                    "phase": "thinking",
                    "state": "end",
                    "duration": _dur(think_start, think_end),
                }
                think_reported = True

            if react and saw_content and not saw_tool and react_content_start is not None:
                write_start = react_content_start
                write_end = stream_end
                yield {"type": "phase", "phase": "writing", "state": "start"}
                yield {
                    "type": "phase",
                    "phase": "writing",
                    "state": "end",
                    "duration": _dur(write_start, write_end),
                }
                write_reported = True
            elif write_start is not None and write_end is None:
                write_end = stream_end
                yield {
                    "type": "phase",
                    "phase": "writing",
                    "state": "end",
                    "duration": _dur(write_start, write_end),
                }
                write_reported = True
            if reasoning:
                yield {
                    "type": "thought",
                    "text": "".join(reasoning),
                    "duration": _dur(think_start, think_end) if (think_start and think_end) else 0.0,
                }

        except OpenRouterError as exc:
            message = exc.message or str(exc)
            elapsed = time.perf_counter() - request_started
            if exc.is_budget or exc.is_rate_limit:
                # Deterministic stop: never retry, never enter ReAct, never loop tools.
                yield {"type": "phase", "phase": "connecting", "state": "error", "duration": elapsed}
                yield {
                    "type": "error",
                    "message": message,
                    "card": exc.card(),
                    "elapsed": elapsed,
                    "fatal": True,
                }
                return
            if native_supported and is_tool_support_error(message):
                yield {"type": "phase", "phase": "connecting", "state": "end", "duration": elapsed}
                if native_recoveries < MAX_NATIVE_RECOVERIES:
                    native_recoveries += 1
                    if native_recoveries == 1:
                        relaxed_native = True
                        recovery = "relaxed provider routing"
                    else:
                        trimmed = [t for t in tools if t["function"]["name"] in NATIVE_RECOVERY_TOOLS]
                        active_tools = trimmed or tools
                        recovery = "minimal core tool set"
                    yield {
                        "type": "notice",
                        "message": f"Tool-call route rejected ({message[:120]}); retrying with {recovery}.",
                    }
                    continue
                if allow_react_fallback:
                    native_supported = False
                    react = True
                    messages[0]["content"] = system + "\n\n" + REACT_SYSTEM.format(tool_names=tool_names)
                    yield {
                        "type": "notice",
                        "message": (
                            f"Native tool calling unavailable after {native_recoveries} retries "
                            f"({message[:160]}); using ReAct tool calls. Use /react off to forbid this."
                        ),
                    }
                    continue
                yield {
                    "type": "error",
                    "fatal": True,
                    "message": (
                        f"Native tool calling rejected by this route after {native_recoveries} retries: "
                        f"{message[:240]} Pick a tool-capable model/provider, or run /react on to allow "
                        "the text-based ReAct tool transport."
                    ),
                    "card": exc.card(),
                    "elapsed": elapsed,
                }
                return
            yield {"type": "phase", "phase": "connecting", "state": "error", "duration": elapsed}
            yield {"type": "error", "message": f"{message} (failed after {elapsed:.1f}s)", "elapsed": elapsed}
            return
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - request_started
            yield {"type": "phase", "phase": "connecting", "state": "error", "duration": elapsed}
            yield {"type": "error", "message": f"{clean_network_error(exc)} (failed after {elapsed:.1f}s)"}
            return

        tool_calls = [tool_acc[i] for i in sorted(tool_acc)]

        if native_supported and tool_calls:
            messages.append({"role": "assistant", "content": "".join(content), "tool_calls": tool_calls})
            budget_hit = False
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or "unknown"
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except Exception:
                    args = {"_raw": raw}
                problem = validate_tool_args(name, args, schemas.get(name))
                if problem is None and tools_used >= tools_per_turn:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or f"call_{name}",
                        "content": json.dumps({"skipped": True, "reason": "tool budget reached", "hint": "Summarize verified findings, evidence refs, dead ends, and next best pivot."}, ensure_ascii=False),
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Tool budget reached ({tools_per_turn} calls). Do not request more tools. Return the best verified answer now, including what worked, what failed, evidence refs, and next pivot.",
                    })
                    yield {"type": "notice", "message": f"Tool budget reached ({tools_per_turn}); requesting final summary."}
                    budget_hit = True
                    break
                if problem is not None:
                    _learn("failed-approach", f"{name} rejected: {problem}", {"tool": name})
                    yield {"type": "error", "fatal": True, "message": problem}
                    return
                guard = _repeat_guard(name, args)
                if guard is not None:
                    if guard.get("type") == "redirect":
                        # Keep OpenAI's protocol valid: each tool_call needs a
                        # tool message even when the duplicate itself is skipped.
                        hint = guard["message"]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.get("id") or f"call_{name}",
                            "content": json.dumps(
                                {"skipped": True, "reason": "duplicate call", "hint": hint},
                                ensure_ascii=False,
                            ),
                        })
                        yield {"type": "notice", "message": hint}
                        continue
                    _learn("failed-approach", str(guard.get("message") or "repeat guard halted"), {"tool": name})
                    yield guard
                    return
                tools_used += 1
                result = ""
                llm_result = ""
                tool_ok = True
                async for event in _execute_tool(registry, name, args):
                    yield event
                    if event.get("type") in ("permission_denied", "hook_blocked"):
                        denials += 1
                    if event.get("type") == "tool_end":
                        result = event.get("result", "")
                        llm_result = event.get("llm_result", result)
                        tool_ok = bool(event.get("ok", True))
                        try:
                            tool_payload = json.loads(result)
                            if isinstance(tool_payload, dict) and (tool_payload.get("denied") or tool_payload.get("blocked")):
                                denials += 1
                        except Exception:
                            pass
                    if denials >= 2:
                        _learn("failed-approach", "two tool actions were denied/blocked by permissions or hooks", {"tool": name})
                        yield {"type": "error", "fatal": True, "message": "Halted: two tool actions were denied/blocked by permissions or hooks. Adjust the request or change /permissions."}
                        return
                _record_repeat_result(name, args, result, tool_ok)
                _learn_tool_result(name, result, tool_ok)
                if should_verify(name, result, tool_ok):
                    needs_verify = True
                    edited_any = True
                messages.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{name}", "content": llm_result or result})
                if name == "submit_and_exit":
                    try:
                        submit_payload = json.loads(result)
                        if isinstance(submit_payload, dict) and submit_payload.get("submitted"):
                            submitted = submit_payload
                    except Exception:
                        pass
                    break
            if submitted is not None:
                summary = str(submitted.get("summary") or "").strip()
                if summary:
                    yield {"type": "delta", "text": summary}
                yield {"type": "done", "submitted": True, "verified": bool(submitted.get("verified"))}
                return
            if budget_hit:
                continue
            if needs_verify or (enable_review and review_rounds == 0):
                async for gate_event in _gates():
                    yield gate_event
            continue

        if react:
            full_content = "".join(content)
            tc = _extract_tool_call(full_content)
            if tc:
                visible = _strip_tool_block(full_content)
                if visible:
                    yield {"type": "delta", "text": visible}
                messages.append({"role": "assistant", "content": full_content})
                name = tc["name"]
                args = tc.get("arguments") or {}
                budget_hit = False
                problem = validate_tool_args(name, args, schemas.get(name))
                if problem is None and tools_used >= tools_per_turn:
                    messages.append({
                        "role": "user",
                        "content": f"Tool budget reached ({tools_per_turn} calls). Do not request more tools. Return the best verified answer now, including what worked, what failed, evidence refs, and next pivot.",
                    })
                    yield {"type": "notice", "message": f"Tool budget reached ({tools_per_turn}); requesting final summary."}
                    continue
                if problem is not None:
                    _learn("failed-approach", f"{name} rejected: {problem}", {"tool": name})
                    yield {"type": "error", "fatal": True, "message": problem}
                    return
                guard = _repeat_guard(name, args)
                if guard is not None:
                    if guard.get("type") == "redirect":
                        hint = guard["message"]
                        messages.append({"role": "user", "content": hint + "\nContinue."})
                        yield {"type": "notice", "message": hint}
                        continue
                    _learn("failed-approach", str(guard.get("message") or "repeat guard halted"), {"tool": name})
                    yield guard
                    return
                tools_used += 1
                result = ""
                llm_result = ""
                tool_ok = True
                async for event in _execute_tool(registry, name, args):
                    yield event
                    if event.get("type") in ("permission_denied", "hook_blocked"):
                        denials += 1
                    if event.get("type") == "tool_end":
                        result = event.get("result", "")
                        llm_result = event.get("llm_result", result)
                        tool_ok = bool(event.get("ok", True))
                        try:
                            tool_payload = json.loads(result)
                            if isinstance(tool_payload, dict) and (tool_payload.get("denied") or tool_payload.get("blocked")):
                                denials += 1
                        except Exception:
                            pass
                    if denials >= 2:
                        _learn("failed-approach", "two tool actions were denied/blocked by permissions or hooks", {"tool": name})
                        yield {"type": "error", "fatal": True, "message": "Halted: two tool actions were denied/blocked by permissions or hooks. Adjust the request or change /permissions."}
                        return
                _record_repeat_result(name, args, result, tool_ok)
                _learn_tool_result(name, result, tool_ok)
                if should_verify(name, result, tool_ok):
                    needs_verify = True
                    edited_any = True
                messages.append({"role": "user", "content": f"Tool result for {name}:\n{llm_result or result}\n\nContinue."})
                if name == "submit_and_exit":
                    try:
                        submit_payload = json.loads(result)
                        if isinstance(submit_payload, dict) and submit_payload.get("submitted"):
                            submitted = submit_payload
                    except Exception:
                        pass
                    if submitted is not None:
                        summary = str(submitted.get("summary") or "").strip()
                        if summary:
                            yield {"type": "delta", "text": summary}
                        yield {"type": "done", "submitted": True, "verified": bool(submitted.get("verified"))}
                        return
                if needs_verify or (enable_review and review_rounds == 0):
                    async for gate_event in _gates():
                        yield gate_event
                continue
            if full_content:
                yield {"type": "delta", "text": full_content}
            if needs_verify or (enable_review and review_rounds == 0):
                async for gate_event in _gates():
                    yield gate_event
                continue
            await _remember(full_content)
            yield {"type": "done"}
            return

        if content:
            pass  # already streamed as delta events
        if needs_verify or (enable_review and review_rounds == 0):
            async for gate_event in _gates():
                yield gate_event
            continue
        await _remember("".join(content))
        yield {"type": "done"}
        return

    yield {"type": "error", "message": f"Agent stopped after {max_steps} steps. Increase with config max_steps if needed."}
