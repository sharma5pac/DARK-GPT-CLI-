"""File snapshots for safe edits and `/undo`.

Snapshots are append-only: nothing is deleted, so the evidence of what changed
is always available. Undo restores the most recent snapshot (or a chosen id).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path.home() / ".config" / "darkgpt" / "undo"


def _path(snapshot_id: str) -> Path:
    return ROOT / f"{snapshot_id}.json"


def save_paths(cwd: str, paths: List[str]) -> str:
    """Snapshot the current bytes of the given paths."""
    files: Dict[str, str] = {}
    for value in paths or []:
        if not value:
            continue
        p = (Path(cwd) / value).resolve()
        if not p.is_file():
            continue
        try:
            files[str(p)] = p.read_text(errors="replace")
        except Exception:
            continue
    if not files:
        return ""
    snapshot_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    entry = {
        "id": snapshot_id,
        "ts": time.time(),
        "cwd": str(Path(cwd).resolve()),
        "files": files,
    }
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        _path(snapshot_id).write_text(json.dumps(entry, indent=2))
    except Exception:
        return ""
    return snapshot_id


def list_snapshots(limit: int = 10) -> List[Dict[str, object]]:
    try:
        entries = []
        for p in sorted(ROOT.glob("*.json"), reverse=True)[:limit]:
            data = json.loads(p.read_text())
            entries.append({
                "id": data.get("id") or p.stem,
                "ts": data.get("ts") or 0,
                "cwd": data.get("cwd") or "",
                "files": list((data.get("files") or {}).keys()),
            })
        return entries
    except Exception:
        return []


def undo(snapshot_id: Optional[str] = None) -> Dict[str, object]:
    """Restore a snapshot. Returns what was restored."""
    try:
        if snapshot_id:
            entry_path = _path(snapshot_id)
        else:
            snapshots = sorted(ROOT.glob("*.json"), reverse=True)
            if not snapshots:
                return {"ok": False, "error": "no snapshots available"}
            entry_path = snapshots[0]
        entry = json.loads(entry_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    restored: List[str] = []
    for filename, content in (entry.get("files") or {}).items():
        try:
            p = Path(filename)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            restored.append(str(p))
        except Exception:
            continue
    return {
        "ok": bool(restored),
        "id": entry.get("id") or entry_path.stem,
        "restored": restored,
    }
