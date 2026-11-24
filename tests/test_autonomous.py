"""Autonomous local-tool routing tests."""

from __future__ import annotations

import asyncio

from agent import autonomous, tool_discovery
from agent.core import ToolRegistry
from utils import permissions


def test_discovery_ranks_local_password_tools(monkeypatch):
    monkeypatch.setattr(
        tool_discovery.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {
            "aircrack-ng", "hashcat", "john", "hash-identifier", "openssl"
        } else None,
    )
    tools = tool_discovery.discover_tools()
    ranked = tool_discovery.rank_tools(
        "password and encryption auditing with aircrack-ng", tools, limit=5
    )
    names = [item["name"] for item in ranked]
    assert names[0] == "aircrack-ng", names
    assert {"hashcat", "john", "hash-identifier", "openssl"}.issubset(set(names)), names
    routes = tool_discovery.build_routes("password and encryption auditing", ranked)
    assert {route["route"] for route in routes} >= {"cli", "script"}
    print("PASS local tool discovery ranks aircrack-ng and alternates")


def test_autonomous_plan_and_multi_route_commands(monkeypatch):
    fake_tools = [{
        "name": "aircrack-ng",
        "path": "/usr/bin/aircrack-ng",
        "category": "password & encryption auditing",
        "tags": ["password", "encryption", "crack"],
        "gui": False,
        "aliases": [],
        "kind": "binary",
        "available": True,
    }]
    monkeypatch.setattr(autonomous.tool_discovery, "discover_tools", lambda extra=None: fake_tools)
    monkeypatch.setattr(
        autonomous,
        "_run_command",
        lambda command, timeout, cwd: {"ok": True, "exit_code": 0, "output": "ok", "duration": 0.01},
    )
    events = []
    payload = autonomous.run(
        "audit wpa password",
        tool="aircrack-ng",
        commands=[
            {"tool": "aircrack-ng", "command": "aircrack-ng --help"},
            {"tool": "hashcat", "command": "hashcat --help"},
        ],
        execute=True,
        probe=False,
        mcp_tool_names=[],
        event=events.append,
    )
    plans = [event for event in events if event["type"] == "autonomous_plan"]
    assert plans and all(plan["goal"] == "audit wpa password" for plan in plans)
    assert plans[-1]["steps"][-1]["status"] == "finished"
    assert payload["ok"] is True
    assert len(payload["executions"]) == 2
    assert all(item["ok"] for item in payload["executions"])
    assert payload["plan"][0]["status"] == "finished"
    print("PASS autonomous plan events and multi-route execution")


def test_autonomous_tool_registration_and_risk():
    registry = ToolRegistry()
    names = {spec["function"]["name"] for spec in registry.specs()}
    assert {"discover_tools", "autonomous_route"}.issubset(names)
    assert permissions.risk_of("discover_tools", {}) == "safe"
    assert permissions.risk_of("autonomous_route", {}) == "safe"
    assert permissions.risk_of("autonomous_route", {"execute": True}) == "shell"
    print("PASS autonomous tools registered and permission-gated")


def test_autonomous_route_emits_plan_event(monkeypatch):
    seen = []

    def fake_run(objective, **kwargs):
        event = kwargs.get("event")
        if event:
            event({
                "type": "autonomous_plan",
                "goal": objective,
                "steps": [{"id": "probe", "label": "Probe", "status": "working"}],
            })
        return {"ok": True, "objective": objective}

    monkeypatch.setattr(autonomous, "run", fake_run)

    async def _run():
        registry = ToolRegistry()
        events = []
        result = await registry.run(
            "autonomous_route",
            {"objective": "audit password"},
            on_event=events.append,
        )
        return result, events

    result, events = asyncio.run(_run())
    assert '"ok": true' in result.lower()
    assert [event["type"] for event in events] == ["autonomous_plan"]
    print("PASS autonomous_route bridges plan events into the agent stream")


def main():
    print("ALL AUTONOMOUS TESTS PASSED")


if __name__ == "__main__":
    main()
