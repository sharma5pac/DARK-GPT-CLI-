"""Temporary git worktrees for isolated review/test runs.

The working tree is never modified. A detached worktree is created at HEAD,
the current diff is applied there, checks run there, and the worktree is
removed. The patch and metadata are preserved under ~/.config/darkgpt/worktrees.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict

from utils import vcs

EVIDENCE = Path.home() / ".config" / "darkgpt" / "worktrees"


def _git(cwd: str, args: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
            errors="replace",
        )
        return proc.returncode == 0, (proc.stdout or "")
    except Exception:
        return False, ""


def create(cwd: str) -> Dict[str, str]:
    root = vcs.root(cwd)
    if not root:
        return {}
    patch = vcs.diff(root)
    if not patch:
        return {}
    try:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
    except Exception:
        return {}
    work_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    patch_path = EVIDENCE / f"{work_id}.patch"
    meta_path = EVIDENCE / f"{work_id}.json"
    location = Path(tempfile.mkdtemp(prefix="darkgpt-review-", dir="/tmp/opencode"))
    shutil.rmtree(location)  # git worktree add requires a non-existent path

    ok, _ = _git(root, ["worktree", "add", "--detach", str(location), "HEAD"])
    if not ok:
        return {}
    patch_path.write_text(patch)
    applied, message = _git(str(location), ["apply", "--whitespace=nowarn", str(patch_path)])
    meta = {
        "id": work_id,
        "ts": time.time(),
        "repo": root,
        "path": str(location),
        "patch": str(patch_path),
        "applied": applied,
        "apply_output": message[:2000],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"id": work_id, "path": str(location), "patch": str(patch_path), "applied": applied}


def cleanup(info: Dict[str, str]) -> None:
    if not info:
        return
    location = info.get("path") or ""
    repo = ""
    try:
        meta_path = EVIDENCE / f"{info.get('id')}.json"
        if meta_path.exists():
            repo = str(json.loads(meta_path.read_text()).get("repo") or "")
    except Exception:
        pass
    if repo and location:
        _git(repo, ["worktree", "remove", "--force", location])
    elif location:
        shutil.rmtree(location, ignore_errors=True)
