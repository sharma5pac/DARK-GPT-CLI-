"""Automatic post-edit verification for coding turns.

Detects the project's real test/lint/build commands from its manifest files and
runs them with hard timeouts. Results are fed back into the agent so a turn is
only finished when the checks are green.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

MAX_OUTPUT = 8000


def _has(cwd: str, *names: str) -> bool:
    return any((Path(cwd) / name).exists() for name in names)


def _scripts(cwd: str) -> Dict[str, object]:
    try:
        data = json.loads((Path(cwd) / "package.json").read_text())
        scripts = data.get("scripts")
        return scripts if isinstance(scripts, dict) else {}
    except Exception:
        return {}


def detect(cwd: str) -> List[str]:
    """Return the check commands that actually apply to this project."""
    cwd = cwd or os.getcwd()
    checks: List[str] = []
    venv_py = Path(cwd) / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else "python3"

    if _has(cwd, "pyproject.toml", "requirements.txt", "setup.py", "tox.ini"):
        if _has(cwd, "tests", "test") or _has(cwd, "pyproject.toml"):
            checks.append(f"{py} -m pytest -q")
        checks.append(f"{py} -m compileall -q .")

    scripts = _scripts(cwd)
    for name in ("test", "lint", "typecheck", "build"):
        if name in scripts and isinstance(scripts[name], str):
            checks.append(f"npm run {name}")

    if _has(cwd, "go.mod"):
        checks += ["go test ./...", "go vet ./..."]
    if _has(cwd, "Cargo.toml"):
        checks += ["cargo test", "cargo clippy -- -D warnings"]
    if _has(cwd, "Makefile"):
        try:
            text = (Path(cwd) / "Makefile").read_text(errors="replace")
            for target in ("test", "lint", "typecheck"):
                if any(line.startswith(f"{target}:") for line in text.splitlines()):
                    checks.append(f"make {target}")
        except Exception:
            pass

    # Preserve order, drop duplicates.
    seen = set()
    return [c for c in checks if not (c in seen or seen.add(c))]


def run(cwd: str, commands: List[str], timeout: int = 180) -> List[Dict[str, object]]:
    """Run checks synchronously (callers should use asyncio.to_thread)."""
    results: List[Dict[str, object]] = []
    for command in commands or []:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
            )
            output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
            code = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            output = f"timed out after {timeout}s"
            code = -1
        except Exception as exc:  # noqa: BLE001
            output = str(exc)
            code = -1
        output = output.strip()
        if len(output) > MAX_OUTPUT:
            output = output[-MAX_OUTPUT:]
        results.append({"command": command, "exit_code": code, "ok": code == 0, "output": output})
    return results


def format_results(results: List[Dict[str, object]]) -> str:
    if not results:
        return "No project checks detected; nothing to verify."
    lines = ["Verification results:"]
    for item in results:
        state = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"[{state}] {item.get('command')} (exit {item.get('exit_code')})")
        out = str(item.get("output") or "").strip()
        if out:
            lines.append(out)
    return "\n".join(lines)
