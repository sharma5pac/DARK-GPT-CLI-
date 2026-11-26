"""Strategic advisor model (fast executor + higher-level reviewer).

The advisor is deliberately *not* in the main token stream. It is consulted
only at decision points (start of a plan, a pivot after failed routes, or before
a report), runs in a worker thread with a hard timeout, and its advice arrives
as a compact event plus a system note on the next model turn.

Keys are read through the existing provider helpers and never logged or stored.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from utils import appconfig, get_response

VERSION = 1
ROOT = Path.home() / ".config" / "darkgpt" / "memory"
LATEST = ROOT / "advisor_latest.json"
HISTORY = ROOT / "advisor_history.jsonl"

ADVISOR_SYSTEM = (
    "You are the strategic advisor to an autonomous penetration-testing and engineering agent. "
    "You are given a short snapshot of its current goal, observations and failed routes. "
    "Return concise, actionable strategic guidance only: the single highest-value next move, "
    "which assumptions to drop, and what evidence would prove or disprove the leading hypothesis. "
    "Prefer minimal-impact, reversible, authorized actions. Never claim a result that is not "
    "evidenced. Keep the answer under 220 words with no preamble."
)


def enabled() -> bool:
    return bool(appconfig.get("advisor_enabled", False))


def points() -> List[str]:
    raw = appconfig.get("advisor_points", ["pivot"])
    if isinstance(raw, str):
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if isinstance(raw, list):
        return [str(p).strip().lower() for p in raw if str(p).strip()]
    return ["pivot"]


def wanted(point: str) -> bool:
    return enabled() and str(point).strip().lower() in points()


def _resolve_route(model: str = "", provider: str = "") -> Dict[str, str]:
    """Pick advisor provider/model without ever exposing key material."""
    chosen_provider = str(provider or "").strip().lower()
    chosen_model = str(model or "").strip()
    if chosen_model and not chosen_provider:
        chosen_provider = get_response.get_provider(chosen_model)
    if not chosen_model:
        chosen_model = str(appconfig.get("advisor_model") or "").strip()
    if chosen_model and not chosen_provider:
        chosen_provider = get_response.get_provider(chosen_model)
    current = get_response.current_provider()
    if not chosen_provider:
        alternate = "nvidia" if current == "openrouter" else "openrouter"
        chosen_provider = alternate if get_response.get_api_key(provider=alternate) else current
    if not chosen_model:
        chosen_model = str(appconfig.get(f"model_{chosen_provider}") or "").strip()
        if not chosen_model or chosen_model.startswith("openrouter/"):
            chosen_model = str(get_response.PROVIDERS[chosen_provider]["default_model"] or "")
    if chosen_provider not in get_response.PROVIDERS:
        chosen_provider = current
    return {
        "provider": chosen_provider,
        "label": str(get_response.PROVIDERS[chosen_provider]["label"]),
        "model": chosen_model or str(get_response.PROVIDERS[chosen_provider]["default_model"]),
        "url": str(get_response.chat_url(chosen_provider)),
    }


def consult(
    question: str,
    context: str = "",
    point: str = "pivot",
    model: str = "",
    provider: str = "",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Blocking consultation. Returns a redacted result dictionary."""
    route = _resolve_route(model, provider)
    limit = int(timeout or appconfig.get("advisor_timeout", 25) or 25)
    limit = max(5, min(90, limit))
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "ok": False,
        "point": point,
        "provider": route["provider"],
        "label": route["label"],
        "model": route["model"],
        "advice": "",
        "error": "",
        "duration": 0.0,
        "ts": time.time(),
    }
    try:
        headers = get_response.auth_headers(provider=route["provider"])
        payload = {
            "model": route["model"],
            "messages": [
                {"role": "system", "content": ADVISOR_SYSTEM},
                {"role": "user", "content": (
                    f"Decision point: {point}\n\nQuestion:\n{str(question or '').strip()[:4000]}\n\n"
                    f"Context:\n{str(context or '').strip()[:6000]}"
                ).strip()},
            ],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 700,
        }
        resp = requests.post(route["url"], headers=headers, json=payload, timeout=(8, limit))
        if resp.status_code != 200:
            detail = ""
            try:
                detail = str(resp.text)[:400]
            except Exception:
                pass
            result["error"] = f"{route['label']} returned HTTP {resp.status_code}: {detail}"
            return result
        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            result["error"] = f"{route['label']} returned no advisor choices"
            return result
        message = choices[0].get("message") or {}
        advice = str(message.get("content") or choices[0].get("text") or "").strip()
        if not advice:
            result["error"] = "advisor returned empty guidance"
            return result
        result["ok"] = True
        result["advice"] = advice[:4000]
    except Exception as exc:  # noqa: BLE001 - advisor must never break the main turn
        result["error"] = f"advisor unavailable: {exc.__class__.__name__}: {str(exc)[:300]}"
    finally:
        result["duration"] = round(max(0.0, time.perf_counter() - started), 3)
    return result


async def consult_async(**kwargs: Any) -> Dict[str, Any]:
    limit = int(kwargs.pop("timeout", None) or appconfig.get("advisor_timeout", 25) or 25)
    try:
        return await asyncio.wait_for(asyncio.to_thread(consult, timeout=limit, **kwargs), timeout=limit + 2)
    except asyncio.TimeoutError:
        return {
            "ok": False, "point": kwargs.get("point", "pivot"), "advice": "",
            "error": f"advisor timeout after {limit}s", "provider": "", "model": "",
            "duration": float(limit), "ts": time.time(),
        }


def _persist(result: Dict[str, Any]) -> None:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = LATEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        tmp.replace(LATEST)
        with HISTORY.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception:
        pass


def remember_advice(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("ok") and result.get("advice"):
        _persist(result)
    return result


def latest() -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(LATEST.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def context_block() -> str:
    """Latest advisor guidance for the system prompt (explicitly unverified)."""
    if not enabled():
        return ""
    item = latest()
    advice = str((item or {}).get("advice") or "").strip()
    if not advice:
        return ""
    return (
        "[STRATEGIC ADVISOR - prior out-of-band guidance, verify against live evidence before relying]\n"
        + advice[:1200]
    )


def history(limit: int = 8) -> List[Dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]
        return rows[-max(0, limit):]
    except Exception:
        return []


def status_lines() -> List[str]:
    route = _resolve_route()
    item = latest()
    advice = str((item or {}).get("advice") or "").strip().replace("\n", " ")[:120]
    return [
        f"Strategic advisor: {'ON' if enabled() else 'OFF'} | points: {', '.join(points()) or 'none'}",
        f"  route: {route['label']} ({route['provider']}) | model: {route['model']}",
        f"  latest: {advice or '(none)'}",
        "  /advisor on|off|now [question] | /advisor model <id>",
    ]


def tool_spec() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "consult_advisor",
            "description": (
                "Consult a second, higher-level strategic model for a short independent opinion "
                "before committing to a costly or irreversible route. Use at pivots, before final "
                "reports, or when multiple hypotheses remain. Runs out-of-band with a timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    }
