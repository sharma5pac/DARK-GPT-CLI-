"""Append-only session checkpoints and safe resume.

A checkpoint captures the full text conversation, a compact summary, git state
and the last prompt. Nothing is ever deleted or overwritten - checkpoints are
an evidence trail as well as a resume mechanism.

Resume replays only user/assistant text into the visible conversation. Tool
calls are never re-executed; the model receives the prior text as history and
starts from the current filesystem state, which is the safe behaviour after a
restart or a crash.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path.home() / ".config" / "darkgpt" / "sessions"
MAX_SESSIONS = 500


def _path(session_id: str) -> Path:
    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        raise ValueError("invalid session id")
    return ROOT / f"{session_id}.json"


def create(
    cwd: str,
    messages: List[Dict[str, Any]],
    *,
    model: str = "",
    label: str = "",
    last_prompt: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a checkpoint and return its metadata."""
    from agent import context as context_util
    from utils import vcs

    session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    turns = [
        {"role": m.get("role"), "content": str(m.get("content") or "")}
        for m in (messages or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
    ]
    entry: Dict[str, Any] = {
        "id": session_id,
        "ts": time.time(),
        "cwd": str(Path(cwd).resolve()),
        "model": model,
        "label": label or "",
        "last_prompt": last_prompt or "",
        "turns": turns,
        "checkpoint": context_util.checkpoint_summary(messages or []),
        "git": {
            "root": vcs.root(cwd),
            "status": vcs.status(cwd)[:2000],
            "diff_stat": vcs.diff_stat(cwd)[:1000],
        },
        "extra": extra or {},
    }
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        path = _path(session_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, indent=2, ensure_ascii=False))
        tmp.replace(path)
    except Exception:
        return {}
    return meta(entry)


def _turn_count(entry: Dict[str, Any], checkpoint: Dict[str, Any]) -> int:
    """Turn count from checkpoint metadata or the entry's own turn list.

    ``checkpoint['turns']`` is an int, while ``entry['turns']`` is a list - and
    ``create()`` returns only the metadata dict. Accept either shape so
    ``summary_text()`` never calls ``len()`` on an int.
    """
    for value in (checkpoint.get("turns"), entry.get("turns")):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return len(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return 0


def meta(entry: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = entry.get("checkpoint") or {}
    return {
        "id": entry.get("id") or "",
        "ts": entry.get("ts") or 0,
        "label": entry.get("label") or "",
        "model": entry.get("model") or "",
        "cwd": entry.get("cwd") or "",
        "turns": _turn_count(entry, checkpoint),
        "last_user": checkpoint.get("last_user") or "",
    }


def list_sessions(cwd: str = "", limit: int = 10) -> List[Dict[str, Any]]:
    try:
        entries: List[Dict[str, Any]] = []
        # Filter by cwd BEFORE slicing, otherwise newer sessions from other
        # directories hide the local ones from the list.
        for path in sorted(ROOT.glob("*.json"), reverse=True):
            try:
                entry = json.loads(path.read_text())
            except Exception:
                continue
            item = meta(entry)
            if cwd and item.get("cwd") and item["cwd"] != str(Path(cwd).resolve()):
                continue
            entries.append(item)
            if len(entries) >= max(0, limit):
                break
        return entries
    except Exception:
        return []


def load(session_id: str) -> Dict[str, Any]:
    try:
        return json.loads(_path(session_id).read_text())
    except Exception:
        return {}


def delete_session(session_id: str) -> bool:
    """Remove one checkpoint file after explicit user confirmation.

    This only touches the checkpoint JSON under ~/.config/darkgpt/sessions; it
    never removes evidence, target data, or reports.
    """
    try:
        path = _path(session_id)
    except ValueError:
        return False
    if not path.exists():
        return False
    path.unlink()
    return True


def resolve(selector: str = "", cwd: str = "") -> Dict[str, Any]:
    """Resolve 'latest', an id, or an id prefix to a session entry."""
    selector = (selector or "").strip()
    if not selector or selector.lower() == "latest":
        sessions = list_sessions(cwd, limit=1)
        return load(sessions[0]["id"]) if sessions else {}
    try:
        direct = load(selector)
        if direct:
            return direct
    except Exception:
        pass
    try:
        for path in sorted(ROOT.glob(f"{selector}*.json"), reverse=True):
            entry = json.loads(path.read_text())
            if entry.get("id", "").startswith(selector):
                return entry
    except Exception:
        pass
    return {}


def replay_messages(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Text-only history safe to hand back to the model."""
    return [
        {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
        for m in (entry.get("turns") or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
    ]


def summary_text(entry: Dict[str, Any]) -> str:
    checkpoint = entry.get("checkpoint") or {}
    git = entry.get("git") or {}
    lines = [
        f"Session {entry.get('id')} | {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(entry.get('ts') or 0)))}",
        f"model at checkpoint: {entry.get('model') or '(unknown)'}",
        f"turns: {_turn_count(entry, checkpoint)}",
    ]
    if entry.get("label"):
        lines.append(f"label: {entry['label']}")
    if git.get("status"):
        lines.append("git status at checkpoint:\n" + str(git["status"])[:800])
    digest = str(checkpoint.get("summary") or "")
    if not digest and entry.get("last_user"):
        digest = f"last prompt: {entry['last_user']}"
    if digest:
        lines.append("\n" + digest[:3000])
    return "\n".join(lines)
