"""Autonomous tool orchestration: discover -> rank -> probe -> execute -> evidence.

This is the runtime that turns "use aircrack-ng for password and encryption
auditing" into a real multi-route workflow: local binaries, Kala wrappers, GUI
routes, and bounded scripts. It emits plan events so the TUI can show
GOAL / WORKING / FINISHED next to the prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent import blackboard, evidence, tool_discovery
from utils import appconfig

EventFn = Optional[Callable[[Dict[str, Any]], None]]


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return default


def _emit(event: EventFn, goal: str, steps: Sequence[Dict[str, Any]]) -> None:
    if event is None:
        return
    try:
        event({
            "type": "autonomous_plan",
            "goal": goal,
            "steps": [dict(step) for step in steps],
            "ts": time.time(),
        })
    except Exception:
        pass


def build_plan(objective: str, routes: Sequence[Dict[str, Any]], command: str = "") -> List[Dict[str, Any]]:
    primary = routes[0]["next"] if routes else "inspect available tools"
    execute_label = command or primary
    return [
        {"id": "discover", "label": "Discover installed tools + Kala wrappers", "status": "finished"},
        {"id": "rank", "label": "Rank routes for the objective", "status": "finished"},
        {"id": "probe", "label": f"Probe: {str(primary)[:100]}", "status": "working"},
        {"id": "execute", "label": ascii_label(execute_label, 100), "status": "pending"},
        {"id": "evidence", "label": "Record evidence + verify goal", "status": "pending"},
    ]


def ascii_label(text: Any, limit: int = 120) -> str:
    """Keep plan labels ASCII and bounded for the TUI prompt panel."""
    raw = str(text or "").replace("\r", " ").replace("\n", " ")
    return " ".join(raw.split())[:limit]


def parse_commands(value: Any) -> List[Dict[str, str]]:
    """Normalise the optional multi-route command map.

    Accepts {"tool": "command"}, [{"tool": ..., "command": ...}], or a bare
    command string. This lets the model run several ranked routes in one turn
    without inventing a new protocol.
    """
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [{"tool": "manual", "command": value.strip()}] if value.strip() else []
    if isinstance(value, dict):
        # {"tool": "cmd"} or {"commands": [...]} / {"command": "cmd"}
        nested = value.get("commands")
        if nested not in (None, "", [], {}):
            return parse_commands(nested)
        single = str(value.get("command") or "").strip()
        if single:
            return [{"tool": str(value.get("tool") or "manual"), "command": single}]
        out: List[Dict[str, str]] = []
        for key, item in value.items():
            command = str(item.get("command") or "") if isinstance(item, dict) else str(item)
            if command.strip():
                out.append({"tool": str(item.get("tool") or key) if isinstance(item, dict) else str(key), "command": command.strip()})
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(parse_commands(item))
        return out
    return []


def _probe_one(tool: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    if tool.get("gui") or not tool.get("path"):
        return {
            "tool": tool["name"],
            "route": "gui",
            "ok": True,
            "output": f"GUI route via agent-desktop: {tool['name']}",
        }
    command = f"{tool['path']} --help"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return {
            "tool": tool["name"],
            "route": "cli",
            "command": command,
            "exit_code": int(completed.returncode),
            "ok": True,
            "duration": round(time.perf_counter() - started, 3),
            "output": output[:5000],
        }
    except subprocess.TimeoutExpired:
        return {
            "tool": tool["name"],
            "route": "cli",
            "command": command,
            "ok": False,
            "error": f"probe timed out after {timeout}s",
            "duration": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "tool": tool["name"],
            "route": "cli",
            "command": command,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_probes(tools: Sequence[Dict[str, Any]], timeout: int = 8, workers: int = 3) -> List[Dict[str, Any]]:
    """Probe tool syntax in parallel; failures are data, not fatal errors."""
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(6, workers))) as pool:
        futures = {pool.submit(_probe_one, tool, timeout): tool for tool in tools}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                tool = futures[future]
                results.append({"tool": tool["name"], "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    results.sort(key=lambda x: x.get("tool", ""))
    return results


def _run_command(command: str, timeout: int, cwd: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return {
            "ok": int(completed.returncode) == 0,
            "exit_code": int(completed.returncode),
            "duration": round(time.perf_counter() - started, 3),
            "output": output[:12000],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"command timed out after {timeout}s",
            "duration": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _record(target: str, tool: str, args: Dict[str, Any], result: str, ok: bool, tags: Sequence[str]) -> Dict[str, Any]:
    if not target:
        return {"recorded": False, "reason": "no active target"}
    try:
        entry = evidence.record(
            target,
            tool=tool,
            args=args,
            result=result,
            ok=ok,
            source="autonomous",
            tags=list(tags),
        )
        return {"recorded": True, "evidence_ref": entry.get("id") or entry.get("evidence_ref") or ""}
    except Exception as exc:  # noqa: BLE001
        return {"recorded": False, "error": f"{type(exc).__name__}: {exc}"}


def run(
    objective: str,
    *,
    tool: str = "",
    command: str = "",
    commands: Any = None,
    probe: bool = True,
    execute: bool = False,
    timeout: int = 30,
    limit: int = 5,
    target: str = "",
    mcp_tool_names: Optional[Sequence[str]] = None,
    event: EventFn = None,
) -> Dict[str, Any]:
    """Discover, rank, probe, then execute explicit commands for each route."""
    objective = ascii_label(objective, 500)
    if not objective:
        return {"ok": False, "error": "objective is required"}

    if mcp_tool_names is not None:
        mcp_names = [str(name) for name in mcp_tool_names]
    else:
        try:
            mcp_names = list((appconfig.get("mcp_tool_names") or []))
        except Exception:
            mcp_names = []
    if not mcp_names:
        # Common Kala convenience names; discovery still verifies the tool
        # inventory locally, so this only improves route labeling.
        mcp_names = [
            "kala_password_cracking_hashcat", "kala_password_cracking_john",
            "kala_wireless_aircrack-ng", "kala_web_app_ffuf",
            "kala_reconnaissance_nmap", "kala_database_sqlmap",
        ]

    explicit = [part.strip() for part in str(tool or "").split(",") if part.strip()]
    tools = tool_discovery.discover_tools(explicit)
    ranked = tool_discovery.rank_tools(objective, tools, mcp_tool_names=mcp_names, limit=_clamp(limit, 1, 10, 5))
    if explicit and not ranked:
        # User named a tool; keep it visible even if keyword scoring missed.
        for name in explicit:
            path = next((t["path"] for t in tools if t["name"] == name), "")
            ranked.append({
                "name": name, "path": path, "category": "user-specified",
                "tags": [], "gui": False, "kind": "binary", "available": bool(path),
                "score": 200, "reason": "explicitly requested by user", "mcp_tool": "",
            })
    routes = tool_discovery.build_routes(objective, ranked)
    command = str(command or "").strip()
    command_list = parse_commands(commands)
    if command and not command_list:
        command_list = [{"tool": "manual", "command": command}]
    steps = build_plan(objective, routes, command or (command_list[0]["command"] if command_list else ""))
    if len(command_list) > 1:
        steps[3]["label"] = f"Execute {len(command_list)} explicit routes"
    _emit(event, objective, steps)

    probes: List[Dict[str, Any]] = []
    if probe and ranked:
        probes = run_probes(ranked[:3], timeout=_clamp(timeout, 3, 20, 8))
        steps[2]["status"] = "finished"
        _emit(event, objective, steps)
    elif not probe:
        steps[2]["status"] = "finished"
        steps[2]["label"] = "Probe skipped"
        _emit(event, objective, steps)

    executions: List[Dict[str, Any]] = []
    execution: Dict[str, Any] = {"ran": False, "reason": "execute=false"}
    if execute:
        steps[3]["status"] = "working"
        _emit(event, objective, steps)
        if not command_list:
            execution = {
                "ran": False,
                "error": "refusing to guess a destructive command; pass command or commands after reading --help",
            }
        else:
            for item in command_list[:8]:
                result = _run_command(
                    item["command"],
                    _clamp(timeout, 3, 120, 30),
                    os.getcwd(),
                )
                entry = {"tool": item["tool"], "command": item["command"], "ran": True, **result}
                executions.append(entry)
                if target:
                    _record(
                        target,
                        f"autonomous:{item['tool']}",
                        {"objective": objective, "command": item["command"]},
                        result.get("output") or result.get("error") or "",
                        bool(result.get("ok")),
                        ["autonomous", "route-execution"],
                    )
            execution = {
                "ran": True,
                "count": len(executions),
                "ok": bool(executions) and all(bool(e.get("ok")) for e in executions),
                "results": executions,
                "command": command or (executions[0]["command"] if executions else ""),
            }
        steps[3]["status"] = "finished" if execution.get("ok") else "pending"
        _emit(event, objective, steps)

    steps[4]["status"] = "working"
    _emit(event, objective, steps)
    target = str(target or appconfig.get("active_target") or "").strip()
    payload = {
        "objective": objective,
        "discovered": [{"name": t["name"], "path": t["path"], "category": t["category"]} for t in tools],
        "ranked": ranked,
        "routes": routes,
        "probes": probes,
        "execution": execution,
        "executions": executions,
        "target": target,
        "plan": steps,
        "note": "Probes are bounded. Destructive/active commands require explicit command(s) after reading --help.",
    }
    recorded = _record(
        target, "autonomous_toolkit",
        {"objective": objective, "tool": tool, "command": command, "commands": command_list},
        json.dumps(payload, ensure_ascii=False),
        bool(execution.get("ok", True)),
        ["autonomous", "tool-discovery"],
    )
    payload["evidence"] = recorded
    if target and recorded.get("evidence_ref"):
        try:
            blackboard.post(
                target,
                "fact",
                f"Objective '{objective[:120]}' routed through: {tool_discovery.inventory_summary(ranked)}",
                source="autonomous",
                refs=[str(recorded["evidence_ref"])],
                meta={"routes": [r["route"] for r in routes]},
            )
        except Exception:
            pass
    steps[4]["status"] = "finished"
    _emit(event, objective, steps)
    payload["ok"] = True
    return payload
