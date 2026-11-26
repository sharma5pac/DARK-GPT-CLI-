"""Persistent DarkGPT TUI configuration (model, mode, tool toggles)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".config" / "darkgpt"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULTS: Dict[str, Any] = {
    "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    "provider": "openrouter",
    "mode": "Build",
    "disabled_tools": [],
    "disabled_servers": [],
    "active_skills": [],
    "mcp_display": "background",
    "verify_enabled": True,
    "memory_enabled": True,
    "self_learning_enabled": True,
    "active_target": "",
    "plan_enabled": False,
    "review_enabled": False,
    "worktree_enabled": True,
    "lsp_enabled": True,
    "subagents_enabled": True,
    "subagent_concurrency": 2,
    "permission_mode": "full",
    "hooks_enabled": True,
    "checkpoint_enabled": True,
    "scripts_enabled": True,
    "allow_react_fallback": True,
    "cheap_model": "",
    # Deferred tool loading: advertise a small always-on set plus tool_search.
    "deferred_tools_enabled": True,
    "tool_search_top_k": 6,
    # Cross-session semantic memory.
    "semantic_memory_enabled": True,
    "semantic_recall_k": 5,
    # Strategic advisor runs out-of-band at decision points. Off by default so
    # the fast executor never pays for a second model unless the user opts in.
    "advisor_enabled": False,
    "advisor_points": ["pivot"],
    "advisor_model": "",
    "advisor_timeout": 25,
    # OWASP agentic hardening: warn annotates untrusted output; enforce also
    # redacts high-confidence injections and quarantines poisoned MCP tools.
    "agentic_guard_mode": "warn",
    "mcp_poison_quarantine": True,
    # GUI replay / action verification.
    "replay_enabled": True,
    "replay_auto_verify": True,
    "replay_max_steps": 40,
    # Provider auto-failover. Never bypasses a quota by reusing the same
    # provider/route; it moves to a different configured backend/model.
    "failover_enabled": True,
    "failover_max_routes": 3,
    "failover_providers": ["nvidia", "openrouter"],
    "failover_models": [],
}


def load_config() -> Dict[str, Any]:
    data = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            raw = json.loads(CONFIG_PATH.read_text())
            if isinstance(raw, dict):
                data.update(raw)
    except Exception:
        pass
    if not isinstance(data.get("disabled_tools"), list):
        data["disabled_tools"] = []
    if not isinstance(data.get("disabled_servers"), list):
        data["disabled_servers"] = []
    if not isinstance(data.get("active_skills"), list):
        data["active_skills"] = []
    for key in ("advisor_points", "failover_providers", "failover_models"):
        value = data.get(key)
        if isinstance(value, str):
            data[key] = [p.strip() for p in value.split(",") if p.strip()]
        elif not isinstance(value, list):
            data[key] = list(DEFAULTS.get(key) or [])
    return data


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.replace(CONFIG_PATH)


def get(key: str, default: Any = None) -> Any:
    return load_config().get(key, default)


def set_value(key: str, value: Any) -> None:
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)


def update(**values: Any) -> Dict[str, Any]:
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)
    return cfg
