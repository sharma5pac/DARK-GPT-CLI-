"""Self-written helper scripts: the model writes a real script, we run it.

The agent can author a Python or bash script for a one-off job (bulk refactor,
data extraction, code generation, test harness, ...) and execute it immediately.
Scripts are written to a dedicated cache directory and are **never deleted** -
they are part of the evidence trail and can be inspected, re-run or committed.

Everything is bounded: source size, runtime, argument count and output length.
The caller is responsible for permission/hook gating through the tool registry.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path.home() / ".cache" / "darkgpt" / "scripts"
MAX_CODE_CHARS = 60_000
MAX_OUTPUT = 12_000
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_ARGS = 32

LANGUAGES = {
    "python": (".py",),
    "py": (".py",),
    "bash": (".sh",),
    "sh": (".sh",),
}


class ScriptError(ValueError):
    """Bad script request - surfaced to the model as a tool error."""


def normalise_language(value: str) -> str:
    lang = str(value or "python").strip().lower()
    if lang not in LANGUAGES:
        raise ScriptError(f"unsupported script language '{lang}'; use python or bash")
    return "python" if lang in ("python", "py") else "bash"


def prepare(language: str, code: str) -> Tuple[Path, List[str]]:
    """Write the script to disk and return ``(path, command_argv)``."""
    lang = normalise_language(language)
    text = str(code or "")
    if not text.strip():
        raise ScriptError("empty script: provide code")
    if len(text) > MAX_CODE_CHARS:
        raise ScriptError(f"script too large ({len(text)} chars, limit {MAX_CODE_CHARS})")
    try:
        SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(f"cannot create script directory {SCRIPT_DIR}: {exc}") from exc

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = SCRIPT_DIR / f"{stamp}-{uuid.uuid4().hex[:8]}{LANGUAGES[lang][0]}"
    if not text.endswith("\n"):
        text += "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(f"cannot write script {path}: {exc}") from exc

    if lang == "python":
        command = [sys.executable or "python3", "-u", str(path)]
    else:
        command = ["/bin/bash", str(path)]
    return path, command


def build_argv(language: str, code: str, args: Optional[List[Any]] = None) -> Tuple[Path, List[str]]:
    """Prepare and append bounded CLI arguments."""
    path, command = prepare(language, code)
    supplied = list(args or [])
    if len(supplied) > MAX_ARGS:
        raise ScriptError(f"too many script arguments ({len(supplied)}, limit {MAX_ARGS})")
    command.extend(str(item) for item in supplied)
    return path, command


def run(
    language: str,
    code: str,
    *,
    args: Optional[List[Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str = "",
) -> Dict[str, Any]:
    """Write and run a script synchronously. Never raises."""
    started = time.perf_counter()
    try:
        bounded = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    except (TypeError, ValueError):
        bounded = DEFAULT_TIMEOUT
    try:
        path, command = build_argv(language, code, args)
    except ScriptError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        proc = subprocess.run(
            command,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded,
        )
        output = ((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")).strip()
        exit_code = int(proc.returncode)
        note = ""
    except subprocess.TimeoutExpired:
        output, exit_code, note = f"[timed out after {bounded}s]", -1, f" timed out after {bounded}s"
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"script execution failed: {exc}"[:MAX_OUTPUT],
            "script_path": str(path),
            "language": normalise_language(language),
        }

    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + f"\n...[truncated {len(output) - MAX_OUTPUT} chars]"
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "output": output or "(no output)",
        "script_path": str(path),
        "language": normalise_language(language),
        "args": [str(a) for a in (args or [])],
        "duration": max(0.0, time.perf_counter() - started),
        "note": note,
    }


def recent(limit: int = 10) -> List[Dict[str, Any]]:
    """List recently written scripts for the /scripts command."""
    items: List[Dict[str, Any]] = []
    try:
        for path in sorted(SCRIPT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(0, limit)]:
            if not path.is_file():
                continue
            stat = path.stat()
            items.append({
                "path": str(path),
                "language": "python" if path.suffix == ".py" else "bash",
                "bytes": stat.st_size,
                "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            })
    except Exception:
        pass
    return items


def show_script(path: str, limit: int = 4000) -> str:
    try:
        target = Path(path).resolve()
        if SCRIPT_DIR not in target.parents:
            return "refusing to read outside the script directory"
        return target.read_text(errors="replace")[:limit]
    except Exception as exc:  # noqa: BLE001
        return f"cannot read script: {exc}"
