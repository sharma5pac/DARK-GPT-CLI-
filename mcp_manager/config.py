"""MCP configuration loader/writer.

Reads and merges the user config (``~/.config/darkgpt/mcp.json``) with the
project config (``./mcp.json``); project entries win. Writes go to the scope
the caller chooses, so the Add Server form can update either file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

USER_CONFIG_PATH = Path.home() / ".config" / "darkgpt" / "mcp.json"
PROJECT_CONFIG_PATH = Path.cwd() / "mcp.json"


def default_config() -> Dict[str, Any]:
    return {
        "mcpServers": {
            "filesystem-example": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.home())],
                "env": {},
                "disabled": True,
            }
        }
    }


def _read(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def ensure_default_config() -> Path:
    if not USER_CONFIG_PATH.exists() and not PROJECT_CONFIG_PATH.exists():
        _write(USER_CONFIG_PATH, default_config())
    return USER_CONFIG_PATH


def load_config() -> Dict[str, Any]:
    ensure_default_config()
    merged: Dict[str, Any] = {"mcpServers": {}}
    for path in (USER_CONFIG_PATH, PROJECT_CONFIG_PATH):
        data = _read(path)
        merged["mcpServers"].update(servers(data))
    return merged


def servers(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = config.get("mcpServers") or config.get("servers") or {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def config_path(scope: str = "user") -> Path:
    return PROJECT_CONFIG_PATH if scope == "project" else USER_CONFIG_PATH


def save_config(config: Dict[str, Any], scope: str = "user") -> Path:
    path = config_path(scope)
    data = _read(path)
    data["mcpServers"] = servers(config)
    _write(path, data)
    return path


def upsert_server(name: str, cfg: Dict[str, Any], scope: str = "user") -> Path:
    path = config_path(scope)
    data = _read(path)
    current = servers(data)
    current[name] = cfg
    data["mcpServers"] = current
    _write(path, data)
    return path


def remove_server(name: str, scope: str = "user") -> bool:
    path = config_path(scope)
    data = _read(path)
    current = servers(data)
    if name not in current:
        return False
    current.pop(name, None)
    data["mcpServers"] = current
    _write(path, data)
    return True


def get_server(name: str) -> Optional[Dict[str, Any]]:
    return servers(load_config()).get(name)
