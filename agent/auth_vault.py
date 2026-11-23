"""Credential/session vault for DarkGPT.

Stores named auth profiles per target outside LLM context. Raw secrets are kept
in a local 0600 JSON file under ~/.config/darkgpt/vault/; tool outputs return
only profile names/metadata/ref ids, never raw secret values.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

VAULT_ROOT = Path.home() / ".config" / "darkgpt" / "vault"
SENSITIVE = {"password", "token", "cookie", "authorization", "api_key", "apikey", "secret", "session", "bearer", "value"}


def _slug(v: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", (v or "default").strip().lower()).strip("-")[:80] or "default"


def vault_path(target: str) -> Path:
    VAULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(VAULT_ROOT, 0o700)
    except Exception:
        pass
    return VAULT_ROOT / f"{_slug(target)}.json"


def _load(target: str) -> Dict[str, Any]:
    p = vault_path(target)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("profiles", {})
            return data
    except Exception:
        pass
    return {"version": 1, "target": target, "profiles": {}, "updated_at": time.time()}


def _save(target: str, data: Dict[str, Any]) -> None:
    p = vault_path(target)
    data["updated_at"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def _redact_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (profile or {}).items():
        lk = k.lower()
        if any(s in lk for s in SENSITIVE):
            if isinstance(v, str):
                out[k] = f"[REDACTED len={len(v)}]"
            elif isinstance(v, dict):
                out[k] = {kk: f"[REDACTED len={len(str(vv))}]" for kk, vv in v.items()}
            else:
                out[k] = "[REDACTED]"
        else:
            out[k] = v
    return out


def set_profile(target: str, name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    d = _load(target)
    name = _slug(name)
    cur = d["profiles"].get(name) or {}
    merged = dict(cur)
    for k, v in (profile or {}).items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v
    merged["name"] = name
    merged["updated_at"] = time.time()
    d["profiles"][name] = merged
    _save(target, d)
    return {"ok": True, "target": target, "profile": _redact_profile(merged)}


def get_profile(target: str, name: str, include_secrets: bool = False) -> Dict[str, Any]:
    d = _load(target)
    p = d["profiles"].get(_slug(name))
    if not p:
        return {"ok": False, "error": f"profile not found: {name}"}
    return {"ok": True, "target": target, "profile": (p if include_secrets else _redact_profile(p))}


def list_profiles(target: str) -> Dict[str, Any]:
    d = _load(target)
    profiles = []
    for name, p in (d.get("profiles") or {}).items():
        profiles.append({
            "name": name,
            "updated_at": p.get("updated_at"),
            "keys": sorted([k for k in p.keys() if k not in ("name", "updated_at")]),
            "has_cookie": bool(p.get("cookies") or p.get("cookie")),
            "has_auth": bool(p.get("authorization") or p.get("token") or p.get("bearer")),
        })
    profiles.sort(key=lambda x: str(x.get("name")))
    return {"ok": True, "target": target, "count": len(profiles), "profiles": profiles, "path": str(vault_path(target))}


def delete_profile(target: str, name: str) -> Dict[str, Any]:
    d = _load(target)
    name = _slug(name)
    if name in d.get("profiles", {}):
        del d["profiles"][name]
        _save(target, d)
        return {"ok": True, "deleted": name}
    return {"ok": False, "error": f"profile not found: {name}"}


def inject_headers(target: str, profile_name: str, base_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Used by execution harness only; never expose returned secrets to LLM."""
    headers = dict(base_headers or {})
    res = get_profile(target, profile_name, include_secrets=True)
    if not res.get("ok"):
        return headers
    p = res.get("profile") or {}
    auth = p.get("authorization") or p.get("auth")
    if not auth and p.get("bearer"):
        auth = f"Bearer {p['bearer']}"
    if not auth and p.get("token"):
        auth = f"Bearer {p['token']}"
    if auth:
        headers["Authorization"] = str(auth)
    cookie = p.get("cookie")
    cookies = p.get("cookies")
    if isinstance(cookies, dict):
        cookie = "; ".join(f"{k}={v}" for k, v in cookies.items())
    elif isinstance(cookies, list):
        parts = []
        for c in cookies:
            if isinstance(c, dict) and c.get("name"):
                parts.append(f"{c['name']}={c.get('value','')}")
        if parts:
            cookie = "; ".join(parts)
    if cookie:
        headers["Cookie"] = str(cookie)
    for k, v in (p.get("headers") or {}).items():
        if v is not None:
            headers[str(k)] = str(v)
    return headers
