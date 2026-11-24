"""Cached first-launch inventory of local tools, GUI apps, skills and folders.

The scan is intentionally bounded and disk-backed:

* first launch (or stale cache) builds the inventory in a worker thread;
* later launches load the JSON cache immediately;
* the compact summary is injected into the agent system prompt;
* ``/tools-scan refresh`` forces a rebuild.

Nothing is uploaded and no files are deleted. Only executable metadata, .desktop
app metadata, skill bundle names, and shallow folder entry counts are recorded.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent import tool_discovery
from utils import appconfig

CACHE_PATH = Path.home() / ".cache" / "darkgpt" / "tool_inventory.json"
VERSION = 3
DEFAULT_MAX_AGE = 86400
MAX_BINARIES = 4000
MAX_APPS = 400
MAX_FOLDERS = 40


def _now() -> float:
    return time.time()


def _empty() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "scanned_at": 0.0,
        "binary_count": 0,
        "relevant_count": 0,
        "app_count": 0,
        "skill_count": 0,
        "folder_count": 0,
        "binaries": [],
        "relevant": [],
        "apps": [],
        "skills": [],
        "folders": [],
    }


def load() -> Dict[str, Any]:
    """Return the cached inventory, or an empty inventory on any error."""
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return _empty()
    base = _empty()
    base.update(data)
    return base


def is_fresh(data: Optional[Dict[str, Any]] = None, max_age: Optional[int] = None) -> bool:
    data = data if data is not None else load()
    if not data.get("scanned_at"):
        return False
    if max_age is None:
        try:
            max_age = int(appconfig.get("tool_inventory_max_age", DEFAULT_MAX_AGE) or DEFAULT_MAX_AGE)
        except Exception:
            max_age = DEFAULT_MAX_AGE
    if max_age <= 0:
        return False
    return max(0.0, _now() - float(data.get("scanned_at") or 0.0)) <= max_age


def _dedupe_paths(paths: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in paths:
        path = str(raw or "").strip()
        if not path:
            continue
        resolved = str(Path(path).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _catalog_info(name: str) -> Dict[str, Any]:
    entry = tool_discovery._catalog_entry(name)
    if entry.get("category") == "general":
        return {"tags": [], "category": "", "gui": False, "relevant": False}
    return {
        "tags": list(entry.get("tags") or []),
        "category": str(entry.get("category") or ""),
        "gui": bool(entry.get("gui")),
        "relevant": True,
    }


def scan_binaries() -> List[Dict[str, Any]]:
    """Enumerate executable files in PATH and known tool locations."""
    home = Path.home()
    roots = _dedupe_paths([
        *(os.environ.get("PATH") or "").split(os.pathsep),
        str(home / ".local" / "bin"),
        str(home / ".cargo" / "bin"),
        str(home / ".npm-global" / "bin"),
        "/usr/local/bin", "/usr/bin", "/usr/sbin", "/usr/games",
        "/opt/homebrew/bin", "/opt/local/bin", "/snap/bin",
        "/flatpak/exports/bin",
    ])
    found: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        if len(found) >= MAX_BINARIES:
            break
        try:
            entries = sorted(os.scandir(root), key=lambda item: item.name.lower())
        except OSError:
            continue
        for entry in entries:
            if len(found) >= MAX_BINARIES:
                break
            name = str(entry.name or "")
            if not name or name.startswith("."):
                continue
            try:
                if not entry.is_file(follow_symlinks=True):
                    continue
                if not (entry.stat(follow_symlinks=True).st_mode & 0o111):
                    continue
            except OSError:
                continue
            key = name.lower()
            if key in found:
                continue
            info = _catalog_info(name)
            item = {
                "name": name,
                "path": str(entry.path),
                "executable": True,
                "relevant": info["relevant"],
                "category": info["category"],
                "tags": info["tags"],
            }
            found[key] = item
    return sorted(found.values(), key=lambda item: item["name"].lower())


def scan_desktop_apps() -> List[Dict[str, Any]]:
    """Read installed .desktop launchers to discover GUI applications."""
    home = Path.home()
    roots = _dedupe_paths([
        str(home / ".local" / "share" / "applications"),
        "/usr/share/applications",
        "/usr/local/share/applications",
        "/var/lib/snapd/desktop/applications",
        str(home / ".local" / "share" / "flatpak" / "exports" / "share" / "applications"),
        "/var/lib/flatpak/exports/share/applications",
    ])
    apps: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        try:
            paths = sorted(Path(root).glob("*.desktop"), key=lambda p: p.name.lower())
        except OSError:
            continue
        for path in paths[:MAX_APPS]:
            try:
                fields: Dict[str, str] = {}
                for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    if key in ("name", "exec", "tryexec", "categories", "nodisplay", "hidden") and key not in fields:
                        fields[key] = value.strip()
                if fields.get("nodisplay", "").lower() in ("1", "true") or fields.get("hidden", "").lower() in ("1", "true"):
                    continue
                name = fields.get("name") or path.stem
                raw_exec = fields.get("exec") or fields.get("tryexec") or ""
                command = re.sub(r"%[A-Za-z]", "", raw_exec).strip()
                binary = ""
                try:
                    parts = shlex.split(command)
                    binary = parts[0] if parts else ""
                except ValueError:
                    binary = command.split(" ", 1)[0]
                resolved = ""
                if binary:
                    if binary.startswith("/"):
                        resolved = binary if Path(binary).exists() else ""
                    else:
                        resolved = shutil.which(binary) or ""
                info = _catalog_info(name)
                if not info["relevant"] and binary:
                    info = _catalog_info(Path(binary).name)
                item = {
                    "name": name,
                    "desktop_file": str(path),
                    "binary": binary,
                    "path": resolved,
                    "categories": [c for c in fields.get("categories", "").split(";") if c][:12],
                    "gui": True,
                    "relevant": info["relevant"],
                    "category": info["category"] or "application",
                    "tags": info["tags"],
                }
                apps.setdefault(name.lower(), item)
            except OSError:
                continue
    return sorted(apps.values(), key=lambda item: item["name"].lower())[:MAX_APPS]


def scan_skills() -> List[Dict[str, Any]]:
    """List installed DarkGPT skill bundles without reading their contents."""
    root = Path.home() / ".config" / "darkgpt" / "skills"
    try:
        paths = sorted(root.glob("**/SKILL.md"), key=lambda p: str(p).lower())
    except OSError:
        return []
    skills: List[Dict[str, Any]] = []
    for path in paths[:400]:
        try:
            relative = path.parent.relative_to(root).as_posix()
        except ValueError:
            relative = path.parent.name
        skills.append({"name": relative, "path": str(path)})
    return skills


def scan_folders() -> List[Dict[str, Any]]:
    """Shallow-count useful working folders; never walk trees recursively."""
    home = Path.home()
    cwd = os.getcwd()
    candidates = [
        cwd, str(home), str(home / "Desktop"), str(home / "Downloads"),
        str(home / "Documents"), str(home / ".local" / "bin"),
        str(home / ".config" / "darkgpt"), str(home / ".config" / "darkgpt" / "skills"),
        str(home / ".config" / "darkgpt" / "targets"),
        str(home / ".cache" / "darkgpt"),
    ]
    folders: List[Dict[str, Any]] = []
    for raw in _dedupe_paths(candidates):
        path = Path(raw)
        entries = 0
        exists = path.exists()
        if exists and path.is_dir():
            try:
                with os.scandir(path) as iterator:
                    entries = sum(1 for _ in iterator)
            except OSError:
                entries = -1
        folders.append({"path": raw, "exists": exists, "entries": entries})
        if len(folders) >= MAX_FOLDERS:
            break
    return folders


def scan() -> Dict[str, Any]:
    """Build and persist a fresh inventory."""
    binaries = scan_binaries()
    apps = scan_desktop_apps()
    skills = scan_skills()
    folders = scan_folders()
    relevant = [item for item in binaries if item.get("relevant")]
    data = {
        "version": VERSION,
        "scanned_at": _now(),
        "binary_count": len(binaries),
        "relevant_count": len(relevant),
        "app_count": len(apps),
        "skill_count": len(skills),
        "folder_count": len(folders),
        "binaries": binaries[:MAX_BINARIES],
        "relevant": relevant,
        "apps": apps,
        "skills": skills,
        "folders": folders,
    }
    save(data)
    return data


def save(data: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, CACHE_PATH)


def ensure_fresh(force: bool = False, max_age: Optional[int] = None) -> Tuple[Dict[str, Any], bool]:
    """Load cache; scan only on first launch, staleness, or explicit force."""
    cached = load()
    if force or not is_fresh(cached, max_age=max_age):
        return scan(), True
    return cached, False


def _names(items: Sequence[Dict[str, Any]], limit: int) -> List[str]:
    return [str(item.get("name") or "") for item in items[:limit] if item.get("name")]


_REVERSE_TAGS = {
    "firmware", "hardware", "embedded", "jtag", "swd", "spi", "i2c",
    "flash", "rom", "squashfs", "rootfs", "qemu", "emulate", "disassembly",
    "decompile", "logic", "analyzer", "sdr", "serial", "uart", "mcu",
}


def _reverse_tools(items: Sequence[Dict[str, Any]], limit: int = 24) -> List[str]:
    matched = [item for item in items if _REVERSE_TAGS.intersection(set(item.get("tags") or []))]
    return _names(matched, limit)


def startup_summary(data: Optional[Dict[str, Any]] = None, scanned: bool = False) -> str:
    data = data or load()
    label = "FIRST-LAUNCH SCAN" if scanned else "CACHED INVENTORY"
    relevant = _names(data.get("relevant") or [], 14)
    apps = _names([item for item in (data.get("apps") or []) if item.get("relevant")], 8)
    stamp = data.get("scanned_at") or 0.0
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(stamp))) if stamp else "never"
    lines = [
        f"{label}: {data.get('binary_count', 0)} binaries "
        f"({data.get('relevant_count', 0)} relevant), "
        f"{data.get('app_count', 0)} GUI apps, "
        f"{data.get('skill_count', 0)} skill bundles, "
        f"{data.get('folder_count', 0)} work folders | cached {when}",
    ]
    reverse = _reverse_tools(data.get("relevant") or [], 16)
    if relevant:
        lines.append("Relevant tools: " + ", ".join(relevant))
    if reverse:
        lines.append("Hardware/firmware reversing: " + ", ".join(reverse))
    if apps:
        lines.append("Relevant GUI apps: " + ", ".join(apps))
    lines.append("The model can call discover_tools/autonomous_route. /tools-scan refresh rescans.")
    return "\n".join(lines)


def context_summary(data: Optional[Dict[str, Any]] = None, limit: int = 2200) -> str:
    """Compact capability block for the system prompt."""
    data = data if data is not None else load()
    if not data.get("scanned_at"):
        return ""
    relevant = _names(data.get("relevant") or [], 48)
    app_names = _names([item for item in (data.get("apps") or []) if item.get("relevant")], 14)
    skill_names = _names(data.get("skills") or [], 18)
    folders = [str(item.get("path") or "") for item in (data.get("folders") or [])[:10] if item.get("path")]
    lines = [
        "# Local capability inventory (cached first-launch scan)",
        f"Verified: {data.get('binary_count', 0)} binaries, {data.get('relevant_count', 0)} security/relevant tools, "
        f"{data.get('app_count', 0)} GUI apps, {data.get('skill_count', 0)} skill bundles.",
    ]
    reverse = _reverse_tools(data.get("relevant") or [], 28)
    if relevant:
        lines.append("Security tools: " + ", ".join(relevant))
    if reverse:
        lines.append("Hardware/firmware reversing: " + ", ".join(reverse))
    if app_names:
        lines.append("GUI apps: " + ", ".join(app_names))
    if skill_names:
        lines.append("Skills: " + ", ".join(skill_names))
    if folders:
        lines.append("Known folders: " + ", ".join(folders))
    lines.append("Before choosing a route, call discover_tools or autonomous_route and pivot across CLI/Kala/GUI/script.")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[: max(0, limit - 20)] + "\n...[inventory truncated]"
