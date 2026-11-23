"""Small git helpers. Every call is bounded and degrades to empty output."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List

MAX_OUTPUT = 6000


def available() -> bool:
    return bool(shutil.which("git"))


def root(cwd: str = "") -> str:
    ok, out = _git(cwd or os.getcwd(), ["rev-parse", "--show-toplevel"])
    return out.strip() if ok else ""


def _git(cwd: str, args: List[str], timeout: int = 15) -> tuple[bool, str]:
    if not available():
        return False, ""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return proc.returncode == 0, (proc.stdout or "")
    except Exception:
        return False, ""


def status(cwd: str = "") -> str:
    ok, out = _git(cwd, ["status", "--short"])
    return out.strip() if ok else ""


def diff(cwd: str = "", staged: bool = False) -> str:
    args = ["diff", "--cached"] if staged else ["diff"]
    ok, out = _git(cwd, args)
    # Keep the trailing newline: `git apply` rejects a stripped patch.
    return out[:MAX_OUTPUT] if ok else ""


def diff_stat(cwd: str = "") -> str:
    ok, out = _git(cwd, ["diff", "--stat"])
    return out.strip()[:1000] if ok else ""


def changed_files(cwd: str = "") -> List[str]:
    ok, out = _git(cwd, ["diff", "--name-only"])
    if not ok:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()][:100]


def log(cwd: str = "", limit: int = 8) -> str:
    ok, out = _git(cwd, ["log", f"-{max(1, min(limit, 30))}", "--oneline", "--decorate"])
    return out.strip()[:MAX_OUTPUT] if ok else ""


def blame(cwd: str, path: str) -> str:
    ok, out = _git(cwd, ["blame", "-L", "1,120", "--", path])
    return out.strip()[:MAX_OUTPUT] if ok else ""


def snapshot(cwd: str = "") -> str:
    """Compact repo context for the system prompt."""
    cwd = cwd or os.getcwd()
    if not root(cwd):
        return ""
    parts = []
    st = status(cwd)
    stat = diff_stat(cwd)
    if st:
        parts.append("Git status (current changes):\n" + st[:1500])
    if stat:
        parts.append("Uncommitted diff stat:\n" + stat)
    return "\n\n".join(parts)
