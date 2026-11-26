"""GUI replay and action verification through MCP (agent-desktop in particular).

DarkGPT already drives desktop applications through the MCP accessibility
bridge. This module turns those real tool calls into bounded, repeatable flows:

* ``capture`` records each desktop MCP call (with secrets redacted) into a
  process-local draft while the agent is working.
* ``save`` persists the draft or an explicit step list as a named flow.
* ``run`` replays the flow through the live MCP manager and checks per-step
  expectations plus an optional final accessibility-snapshot assertion.

Nothing is deleted: flows are append-only JSON records and every replay gets
its own JSONL result. Replay is deliberately bounded (default 40 steps) so a
poisoned or looping flow cannot drive a desktop forever.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import appconfig

VERSION = 1
ROOT = Path.home() / ".config" / "darkgpt" / "replay"
STORE = ROOT / "flows.json"
LOG = ROOT / "replay.jsonl"

MAX_FLOWS = 200
MAX_LOG = 1000
SECRET_KEY_RE = re.compile(r"(?i)(key|token|secret|password|passwd|cookie|authorization|credential)")
SNAPSHOT_TOOLS = ("desktop_snapshot", "snapshot")

_LOCK = threading.RLock()
_CAPTURE: Optional[Dict[str, Any]] = None


def enabled() -> bool:
    return bool(appconfig.get("replay_enabled", True))


def _max_steps() -> int:
    try:
        return max(1, min(120, int(appconfig.get("replay_max_steps", 40) or 40)))
    except Exception:
        return 40


def _load() -> List[Dict[str, Any]]:
    try:
        rows = json.loads(STORE.read_text())
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    except Exception:
        return []


def _save(flows: Sequence[Dict[str, Any]]) -> None:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(flows[-MAX_FLOWS:]), indent=2, ensure_ascii=False))
        tmp.replace(STORE)
    except Exception:
        pass


def _append_log(row: Dict[str, Any]) -> None:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def redact(value: Any) -> Any:
    """Redact credential-looking args/values; never persist live secrets."""
    if isinstance(value, dict):
        return {
            str(k): ("[redacted]" if SECRET_KEY_RE.search(str(k)) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    text = str(value)
    # Bearer tokens and long opaque strings are not useful in a replay flow.
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer [redacted]", text)
    return text[:4000]


def start_capture(name: str) -> str:
    """Start (or restart) collecting desktop MCP calls for a named flow."""
    global _CAPTURE
    with _LOCK:
        label = str(name or f"flow-{int(time.time())}").strip()
        _CAPTURE = {"name": label, "steps": [], "started": time.time()}
        return label


def capture(tool: str, args: Dict[str, Any], result: Any = None) -> int:
    """Record one live tool call into the active capture draft."""
    with _LOCK:
        if _CAPTURE is None:
            return 0
        name = str(tool or "")
        if not (name.startswith("desktop_") or name.startswith("mcp__agent-desktop__")):
            return 0
        step: Dict[str, Any] = {"tool": name, "args": redact(args or {})}
        if result is not None:
            preview = _text_of(result)
            if preview:
                step["observed"] = preview[:1200]
        _CAPTURE["steps"].append(step)
        return len(_CAPTURE["steps"])


def captured_steps() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(s) for s in (_CAPTURE or {}).get("steps", [])]


def stop_capture() -> List[Dict[str, Any]]:
    global _CAPTURE
    with _LOCK:
        steps = [dict(s) for s in (_CAPTURE or {}).get("steps", [])]
        _CAPTURE = None
        return steps


def _normalize_steps(steps: Any) -> List[Dict[str, Any]]:
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except Exception:
            steps = None
    if not isinstance(steps, (list, tuple)):
        return []
    limit = _max_steps()
    out: List[Dict[str, Any]] = []
    for item in steps[:limit]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("name") or "").strip()
        if not tool:
            continue
        step: Dict[str, Any] = {"tool": tool, "args": redact(item.get("args") or {})}
        if item.get("expect"):
            step["expect"] = str(item["expect"])[:500]
        if item.get("observed"):
            step["observed"] = str(item["observed"])[:1200]
        out.append(step)
    return out


def save(
    name: str,
    steps: Any = None,
    *,
    verify: str = "",
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Persist a replayable GUI flow. Uses the active capture when steps are omitted."""
    if not enabled():
        return {"ok": False, "error": "GUI replay is disabled (/replay on)."}
    label = str(name or "").strip() or f"flow-{int(time.time())}"
    normalized = _normalize_steps(steps) if steps is not None else captured_steps()
    if not normalized:
        return {"ok": False, "error": "no steps to save: start a capture or pass a step list"}
    flows = _load()
    flow = {
        "id": f"rp_{int(time.time()*1000):x}",
        "name": label,
        "ts": time.time(),
        "steps": normalized,
        "verify": str(verify or "")[:500],
        "tags": sorted({str(t).strip() for t in (tags or []) if str(t).strip()})[:12],
    }
    flows.append(flow)
    _save(flows)
    return {"ok": True, "id": flow["id"], "name": label, "steps": len(normalized), "verify": flow["verify"]}


def list_flows(limit: int = 12) -> List[Dict[str, Any]]:
    rows = _load()
    rows.sort(key=lambda f: float(f.get("ts") or 0.0), reverse=True)
    return [
        {
            "id": f.get("id"), "name": f.get("name"), "steps": len(f.get("steps") or []),
            "verify": f.get("verify") or "", "ts": f.get("ts"),
        }
        for f in rows[: max(0, min(100, int(limit or 12)))]
    ]


def get_flow(name_or_id: str) -> Optional[Dict[str, Any]]:
    key = str(name_or_id or "").strip().lower()
    if not key:
        return None
    rows = _load()
    for flow in rows:
        if str(flow.get("id") or "").lower() == key or str(flow.get("name") or "").lower() == key:
            return flow
    return None


def history(limit: int = 8) -> List[Dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
        return rows[-max(0, min(100, int(limit or 8))):]
    except Exception:
        return []


def _text_of(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            parts.append(text if isinstance(text, str) else str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content or "")


def _resolve_call(manager: Any, tool: str) -> Optional[Tuple[str, str]]:
    name = str(tool or "").strip()
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    if "/" in name:
        server, _, rest = name.partition("/")
        return server, rest
    states = getattr(manager, "servers", {}) or {}
    preferred = ("agent-desktop", "kala-mcp-root")
    for server in preferred:
        state = states.get(server)
        if state and any(getattr(t, "name", "") == name for t in (state.tools or [])):
            return server, name
    for server, state in states.items():
        if any(getattr(t, "name", "") == name for t in (state.tools or [])):
            return server, name
    # agent-desktop is still the best default for desktop_* names before meta
    # has finished loading; call_tool will surface a clear error if it is absent.
    if name.startswith("desktop_"):
        return "agent-desktop", name
    return None


async def run(
    manager: Any,
    name_or_id: str,
    *,
    verify_only: bool = False,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Replay a saved flow through live MCP and verify the observed state."""
    if not enabled():
        return {"ok": False, "error": "GUI replay is disabled (/replay on)."}
    flow = get_flow(name_or_id)
    if not flow:
        return {"ok": False, "error": f"no replay flow named '{name_or_id}' (/replay list)."}
    if manager is None:
        return {"ok": False, "error": "MCP manager unavailable; GUI replay needs MCP."}
    steps = list(flow.get("steps") or [])
    if verify_only:
        steps = [s for s in steps if str(s.get("tool") or "").startswith(SNAPSHOT_TOOLS)][-1:]
        if not steps:
            steps = [{"tool": "desktop_snapshot", "args": {}}]
    limit = _max_steps()
    if len(steps) > limit:
        return {"ok": False, "error": f"flow has {len(steps)} steps; limit is {limit}"}
    per_call = max(2, min(120, int(timeout or 20)))
    results: List[Dict[str, Any]] = []
    all_text: List[str] = []
    ok = True
    started = time.perf_counter()

    for index, step in enumerate(steps, 1):
        tool = str(step.get("tool") or "")
        target = _resolve_call(manager, tool)
        if target is None:
            results.append({"step": index, "tool": tool, "ok": False, "error": "tool not found on any MCP server"})
            ok = False
            break
        server, actual = target
        try:
            response = await asyncio.wait_for(
                manager.call_tool(server, actual, redact(step.get("args") or {})),
                timeout=per_call,
            )
            text = redact(_text_of(response))
            all_text.append(text)
            expect = str(step.get("expect") or "")
            matched = True
            if expect:
                try:
                    matched = bool(re.search(expect, text, re.IGNORECASE | re.DOTALL))
                except re.error:
                    matched = expect.lower() in text.lower()
            if not matched:
                ok = False
            results.append({
                "step": index, "tool": f"{server}:{actual}", "ok": matched,
                "preview": text[:500], "expected": expect or None,
            })
            if not matched:
                break
        except Exception as exc:  # noqa: BLE001 - replay must report, never crash the turn
            ok = False
            results.append({
                "step": index, "tool": f"{server}:{actual}", "ok": False,
                "error": f"{exc.__class__.__name__}: {str(exc)[:250]}",
                "hint": "Refresh the accessibility snapshot and re-capture refs before replaying.",
            })
            break

    combined = "\n".join(all_text)
    verdict = "verified" if ok else "unverified"
    expected = str(flow.get("verify") or "")
    if ok and expected:
        try:
            matched = bool(re.search(expected, combined, re.IGNORECASE | re.DOTALL))
        except re.error:
            matched = expected.lower() in combined.lower()
        if not matched:
            verdict = "unverified"
            results.append({"step": len(results) + 1, "tool": "verify", "ok": False, "expected": expected, "preview": combined[:500]})
    duration = round(max(0.0, time.perf_counter() - started), 3)
    record = {
        "ok": verdict == "verified", "flow": flow.get("name"), "id": flow.get("id"),
        "verdict": verdict, "steps": results, "duration": duration, "ts": time.time(),
    }
    _append_log(record)

    target = str(appconfig.get("active_target") or "").strip()
    if target:
        try:
            from agent import evidence as evidence_util

            await asyncio.to_thread(
                evidence_util.record, target, tool="replay_run",
                args={"flow": flow.get("name"), "verify_only": verify_only},
                result=json.dumps(record, ensure_ascii=False)[:8000],
                ok=record["ok"], source="replay_run", tags=["gui", "replay", verdict],
            )
        except Exception:
            pass
    return record


def status_lines() -> List[str]:
    flows = list_flows(3)
    recent = history(3)
    return [
        f"GUI replay: {'ON' if enabled() else 'OFF'} | flows={len(_load())} | max_steps={_max_steps()}",
        "  recent flows: " + (", ".join(f"{f['name']}({f['steps']})" for f in flows) or "none"),
        "  last verdict: " + (str(recent[-1].get("verdict") or "-") if recent else "none"),
        "  /replay | /replay list | /replay run <name> | /replay save <name> <steps.json>",
    ]


def tool_spec() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "replay_run",
            "description": (
                "Save or replay a bounded GUI action flow through the live MCP desktop bridge and "
                "verify the resulting accessibility state. Actions: status, list, capture, save, run, verify. "
                "Use capture before a GUI sequence, then save with an optional regex 'verify' assertion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["status", "list", "capture", "save", "run", "verify"]},
                    "name": {"type": "string", "description": "Flow name or id."},
                    "steps": {"description": "List of {tool,args,expect?} objects for save."},
                    "verify": {"type": "string", "description": "Final regex assertion against snapshot/action output."},
                    "limit": {"type": "integer"},
                },
                "required": ["action"],
            },
        },
    }
