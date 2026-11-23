"""Configurable shell hooks for agent events.

Hooks are defined in JSON at ``~/.config/darkgpt/hooks.json`` and/or
``<repo>/.darkgpt/hooks.json``::

    {
      "pre_edit": ["python tools/check_style.py"],
      "post_shell": [{"command": "notify-send 'DarkGPT ran $DARKGPT_TOOL'", "timeout": 3}]
    }

A non-zero exit from a ``pre_*`` hook blocks the tool (its output becomes the
reason). ``post_*`` hooks never block. Every hook is bounded by a timeout and
cannot raise into the agent.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_TIMEOUT = 5
MAX_OUTPUT = 3000
EVENTS = {
    "pre_tool", "post_tool", "pre_edit", "post_edit", "pre_shell", "post_shell",
    "user_prompt_submit", "stop",
}


@dataclass
class HookResult:
    event: str
    allowed: bool = True
    outputs: List[str] = field(default_factory=list)
    blocked_by: str = ""

    def reason(self) -> str:
        return " | ".join(self.outputs[-2:]) or self.blocked_by or "pre-hook rejected the action"


def _paths(cwd: str) -> List[Path]:
    try:
        from utils import vcs

        root = Path(vcs.root(cwd) or cwd).resolve()
    except Exception:
        root = Path(cwd).resolve()
    return [
        Path.home() / ".config" / "darkgpt" / "hooks.json",
        root / ".darkgpt" / "hooks.json",
    ]


def load(cwd: str = "") -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {event: [] for event in EVENTS}
    for path in _paths(cwd or os.getcwd()):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for event, entries in data.items():
            if event not in EVENTS or not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, str):
                    merged[event].append({"command": entry, "timeout": DEFAULT_TIMEOUT})
                elif isinstance(entry, dict) and entry.get("command"):
                    merged[event].append({
                        "command": str(entry["command"]),
                        "timeout": max(1, min(int(entry.get("timeout", DEFAULT_TIMEOUT)), 30)),
                    })
    return merged


def configure(cwd: str = "", **events: List[Any]) -> Path:
    """Write/replace hooks config atomically (preserving unknown keys)."""
    path = Path(cwd or os.getcwd()).resolve() / ".darkgpt" / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    try:
        existing = json.loads(path.read_text())
    except Exception:
        pass
    if not isinstance(existing, dict):
        existing = {}
    for event, entries in events.items():
        if event in EVENTS:
            existing[event] = entries
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    tmp.replace(path)
    return path


def run(event: str, context: Dict[str, Any] | None = None, cwd: str = "") -> HookResult:
    """Run all hooks for ``event``. Never raises."""
    result = HookResult(event=event)
    if event not in EVENTS:
        return result
    ctx = context or {}
    env = dict(os.environ)
    env.update({
        "DARKGPT_EVENT": event,
        "DARKGPT_CWD": str(cwd or os.getcwd()),
        "DARKGPT_TOOL": str(ctx.get("tool") or ""),
        "DARKGPT_PATH": str(ctx.get("path") or ""),
        "DARKGPT_COMMAND": str(ctx.get("command") or ctx.get("prompt") or ""),
    })
    for hook in load(cwd).get(event, []):
        command = str(hook.get("command") or "")
        timeout = int(hook.get("timeout", DEFAULT_TIMEOUT))
        try:
            proc = subprocess.run(
                command, shell=True, cwd=cwd or os.getcwd(), env=env,
                capture_output=True, text=True, timeout=timeout, executable="/bin/bash",
                errors="replace",
            )
            output = ((proc.stdout or "") + (proc.stderr or "")).strip()[:MAX_OUTPUT]
            if output:
                result.outputs.append(output)
            if event.startswith("pre_") and proc.returncode != 0:
                result.allowed = False
                result.blocked_by = f"{command[:80]} (exit {proc.returncode})"
                return result  # first blocking hook wins
        except subprocess.TimeoutExpired:
            if event.startswith("pre_"):
                result.allowed = False
                result.blocked_by = f"{command[:80]} timed out after {timeout}s"
                return result
            result.outputs.append(f"{command[:80]} timed out")
        except Exception as exc:  # noqa: BLE001
            result.outputs.append(f"{command[:80]}: {exc}"[:MAX_OUTPUT])
    return result
