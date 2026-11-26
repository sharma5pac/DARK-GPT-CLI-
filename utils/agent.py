"""OpenCode-style agent loop for DarkGPT.

The model can call tools; tool results are fed back until the model produces a
final answer. Events are yielded as dicts for SSE/web and CLI rendering.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import requests

from utils.get_response import OpenRouterError, auth_headers, chat_url, parse_error_response
from utils.search import format_results, web_search

MAX_TOOL_OUTPUT = 12000
MAX_STEPS = 12
DEFAULT_AGENT_MODEL = "nvidia/nemotron-3.5-lightning:free"


def _clamp(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[truncated {len(text) - limit} chars]"


def _safe_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return p.resolve()


# --------------------------------------------------------------------------- tools

def tool_shell(command: str, timeout: int = 120) -> Dict:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=min(max(timeout, 1), 300),
        executable="/bin/bash",
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return {"exit_code": proc.returncode, "output": _clamp(out.strip())}


def tool_read(path: str, offset: int = 1, limit: int = 400) -> Dict:
    p = _safe_path(path)
    if not p.exists() or not p.is_file():
        return {"error": f"File not found: {p}"}
    lines = p.read_text(errors="replace").splitlines()
    start = max(offset - 1, 0)
    chunk = lines[start : start + max(limit, 1)]
    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(chunk)]
    return {"path": str(p), "content": _clamp("\n".join(numbered)), "total_lines": len(lines)}


def tool_write(path: str, content: str) -> Dict:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes": len(content.encode("utf-8"))}


def tool_edit(path: str, old: str, new: str) -> Dict:
    p = _safe_path(path)
    if not p.exists() or not p.is_file():
        return {"error": f"File not found: {p}"}
    text = p.read_text(errors="replace")
    if old not in text:
        return {"error": "old text not found"}
    p.write_text(text.replace(old, new, 1))
    return {"path": str(p), "ok": True}


def tool_ls(path: str = ".") -> Dict:
    p = _safe_path(path)
    if not p.exists():
        return {"error": f"Not found: {p}"}
    if p.is_file():
        return {"path": str(p), "type": "file"}
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:200]:
        entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
    return {"path": str(p), "entries": entries}


def tool_grep(pattern: str, path: str = ".", glob: str = "") -> Dict:
    cmd = ["grep", "-RIn", "--exclude-dir=.git", "--exclude-dir=node_modules", "--exclude-dir=.venv"]
    if glob:
        cmd += [f"--include={glob}"]
    cmd += [pattern, path]
    proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
    out = (proc.stdout or proc.stderr or "").strip()
    return {"output": _clamp(out or "(no matches)")}


def tool_glob(pattern: str, path: str = ".") -> Dict:
    base = _safe_path(path)
    matches = [str(p.relative_to(base)) for p in base.glob(pattern)][:200]
    return {"matches": matches, "count": len(matches)}


def tool_web_search(query: str, max_results: int = 5) -> Dict:
    results = web_search(query, max_results=max_results)
    return {"results": results, "formatted": format_results(results)}


def tool_autonomous_plan(objective: str, tool: str = "", limit: int = 4) -> Dict:
    """Plan-only local-tool route discovery; never executes a command.

    This gives the web/SSE agent the same GOAL/WORKING/FINISHED plan event the
    TUI gets from ``autonomous_route``, without bypassing TUI permission gates.
    """
    try:
        from agent import autonomous, tool_discovery

        limit = max(1, min(int(limit or 4), 10))
        explicit = [p.strip() for p in str(tool or "").split(",") if p.strip()]
        found = tool_discovery.discover_tools(explicit)
        ranked = tool_discovery.rank_tools(str(objective or ""), found, limit=limit)
        routes = tool_discovery.build_routes(str(objective or ""), ranked)
        steps = autonomous.build_plan(str(objective or ""), routes)
        return {"ok": True, "goal": str(objective or ""), "steps": steps, "routes": routes}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "goal": str(objective or ""), "steps": []}


TOOLS = {
    "shell": {
        "desc": "Run a bash command in the current working directory. Use for git, tests, builds, system tools.",
        "params": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]},
        "fn": tool_shell,
    },
    "read": {
        "desc": "Read a text file with line numbers.",
        "params": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]},
        "fn": tool_read,
    },
    "write": {
        "desc": "Write content to a file, creating parent dirs.",
        "params": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        "fn": tool_write,
    },
    "edit": {
        "desc": "Replace first occurrence of old text with new text in a file.",
        "params": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]},
        "fn": tool_edit,
    },
    "ls": {
        "desc": "List directory entries.",
        "params": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
        "fn": tool_ls,
    },
    "grep": {
        "desc": "Search file contents with grep -RIn.",
        "params": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]},
        "fn": tool_grep,
    },
    "glob": {
        "desc": "Find files by glob pattern.",
        "params": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]},
        "fn": tool_glob,
    },
    "web_search": {
        "desc": "Search the web for current information and sources.",
        "params": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]},
        "fn": tool_web_search,
    },
    "autonomous_plan": {
        "desc": "Build a live GOAL/WORKING/FINISHED plan from local tool discovery and ranked routes. Planning only: it never executes commands.",
        "params": {"type": "object", "properties": {"objective": {"type": "string"}, "tool": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["objective"]},
        "fn": tool_autonomous_plan,
    },
}


def tool_specs() -> List[Dict]:
    return [
        {"type": "function", "function": {"name": name, "description": spec["desc"], "parameters": spec["params"]}}
        for name, spec in TOOLS.items()
    ]


def run_tool(name: str, args: Dict) -> str:
    spec = TOOLS.get(name)
    if not spec:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = spec["fn"](**(args or {}))
    except subprocess.TimeoutExpired:
        result = {"error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        result = {"error": str(exc)}
    return json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------- loop

SYSTEM = """You are DarkGPT Agent, a local coding/ops assistant running on the user's machine.
You can use tools to inspect files, edit code, run commands, search, and browse the web.
Be direct and technical. Show what you did. If a command is risky/destructive, say so clearly.
When using web search, cite source URLs. Prefer small, verifiable steps."""


def _chat(payload: Dict, timeout: int = 180) -> Dict:
    headers = auth_headers()
    r = requests.post(chat_url(), headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise parse_error_response(r)
    data = r.json()
    if data.get("error"):
        raise OpenRouterError(str(data["error"]))
    return data


def agent_stream(prompt: str, history: Optional[List[Dict]] = None, model: str = DEFAULT_AGENT_MODEL) -> Iterator[Dict]:
    messages: List[Dict] = [{"role": "system", "content": SYSTEM}]
    for item in history or []:
        role = "assistant" if item.get("role") in ("assistant", "bot") else "user"
        content = (item.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    yield {"type": "meta", "model": model, "agent": True}

    for step in range(MAX_STEPS):
        payload = {"model": model, "messages": messages, "tools": tool_specs(), "tool_choice": "auto"}
        try:
            data = _chat(payload)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": str(exc)}
            return

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        if tool_calls:
            # keep assistant tool-call message in history
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            if content.strip():
                yield {"type": "delta", "text": content}
            for tc in tool_calls:
                fn = (tc.get("function") or {})
                name = fn.get("name") or "unknown"
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {"_raw": raw_args}
                yield {"type": "tool_start", "name": name, "args": args}
                result = run_tool(name, args)
                if name == "autonomous_plan":
                    # Web/SSE parity with the TUI's live plan event.
                    try:
                        parsed = json.loads(result)
                    except Exception:
                        parsed = {}
                    yield {
                        "type": "autonomous_plan",
                        "goal": str(parsed.get("goal") or str(args.get("objective") or "")),
                        "steps": [dict(s) for s in (parsed.get("steps") or []) if isinstance(s, dict)],
                    }
                yield {"type": "tool_end", "name": name, "result": result[:4000]}
                messages.append({"role": "tool", "tool_call_id": tc.get("id", f"call_{step}_{name}"), "content": result})
            continue

        # final answer - stream not available here; emit as one delta for simplicity
        if content:
            yield {"type": "delta", "text": content}
        yield {"type": "done"}
        return

    yield {"type": "error", "message": f"Agent stopped after {MAX_STEPS} steps."}
