"""Universal neutral router: read-only decisions and gated single-tool execution."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agent import neutral_router as router
from agent.core import ToolRegistry


def _specs():
    return ToolRegistry().local_specs()


def _skill(name, description="", triggers=(), tools=()):
    return SimpleNamespace(
        name=name,
        description=description,
        triggers=list(triggers),
        tools=list(tools),
        path=Path("/tmp/skills") / name / "SKILL.md",
    )


def test_decide_is_read_only_and_prefers_actionable_tool():
    decision = router.decide(
        "list files in this directory",
        _specs(),
        [_skill("math-attack-analysis", triggers=["entropy"], tools=["math_attack"])],
    )
    assert decision["decided"] is True
    assert decision["primary"]["name"] == "ls"
    assert decision["executed"] is False
    assert decision["can_execute"] is True
    assert "spec" not in decision["primary"]
    assert "NEUTRAL ROUTER" in decision["text"]
    assert "executed: no" in decision["text"]


def test_decide_keeps_skill_as_context_without_stealing_execution():
    decision = router.decide(
        "check session token entropy",
        _specs(),
        [_skill("math-attack-analysis", triggers=["entropy"], tools=["math_attack"])],
    )
    assert decision["primary"]["name"] == "math_attack"
    assert decision["skill"]["name"] == "math-attack-analysis"
    assert decision["primary"]["name"] != decision["skill"]["name"]
    assert decision["next_action"] == "call_tool"


def test_decide_explicit_skill_loads_the_skill():
    decision = router.decide(
        "use math-attack-analysis skill",
        _specs(),
        [_skill("math-attack-analysis", triggers=["entropy"])],
    )
    assert decision["primary"]["kind"] == "skill"
    assert decision["next_action"] == "load_skill"
    assert decision["can_execute"] is True


def test_decide_reports_missing_arguments_instead_of_guessing():
    decision = router.decide("find admin panels on the target", _specs())
    assert decision["primary"]["name"] == "browser"
    assert decision["can_execute"] is False
    assert "actions" in decision["missing"]
    assert decision["next_action"] == "provide_args"


def test_decide_asks_when_no_capability_matches():
    decision = router.decide("zzzz qqqq", [], scope="tools")
    assert decision["decided"] is False
    assert decision["confidence"] == "low"
    assert decision["question"]
    assert "NOT DECIDED" in decision["text"]


def test_control_plane_tools_are_not_auto_executed():
    decision = router.decide(
        "ask_question",
        _specs(),
        provided_args={"question": "Pick one", "options": ["a", "b"]},
    )
    assert decision["primary"]["name"] == "ask_question"
    assert decision["can_execute"] is False
    assert decision["next_action"] == "blocked"


def test_router_executes_one_selected_tool_through_the_registry(tmp_path):
    registry = ToolRegistry(permission_mode="full", hooks_enabled=False)
    raw = asyncio.run(
        registry.run(
            "neutral_route",
            {
                "goal": "list files in this directory",
                "args": {"path": str(tmp_path)},
                "execute": True,
            },
        )
    )
    payload = json.loads(raw)
    assert payload["executed"] is True
    assert payload["executed_tool"] == "ls"
    assert payload["executed_ok"] is True
    assert "spec" not in payload["primary"]
    assert "executed: yes" in payload["text"]


def test_router_does_not_execute_when_required_args_are_missing():
    registry = ToolRegistry(permission_mode="full", hooks_enabled=False)
    raw = asyncio.run(
        registry.run(
            "neutral_route",
            {"goal": "find admin panels on the target", "execute": True},
        )
    )
    payload = json.loads(raw)
    assert payload["executed"] is False
    assert payload["execute_blocked"]
    assert "actions" in payload["missing"]
    assert "blocked:" in payload["text"]


def test_legacy_web_router_decides_and_executes_opt_in(tmp_path):
    from utils import agent as legacy

    names = {spec["function"]["name"] for spec in legacy.tool_specs()}
    assert "neutral_route" in names
    decision = legacy.tool_neutral_route("list files in this directory")
    assert decision["primary"]["name"] == "ls"
    assert decision["executed"] is False
    executed = legacy.tool_neutral_route(
        "list files in this directory",
        args={"path": str(tmp_path)},
        execute=True,
    )
    assert executed["executed"] is True
    assert executed["executed_tool"] == "ls"
    assert executed["executed_ok"] is True


def test_tui_neutral_command_parses_scope_and_auto(monkeypatch):
    from tui import app as app_mod

    notes = []
    calls = []

    async def fake_run(self, name, args):
        calls.append((name, args))
        return {"ok": True, "text": "NEUTRAL ROUTER fake"}

    monkeypatch.setattr(app_mod.DarkGPTApp, "_run_core_tool", fake_run)
    monkeypatch.setattr(app_mod.DarkGPTApp, "_system_note", lambda self, text: notes.append(str(text)))
    app = app_mod.DarkGPTApp()

    asyncio.run(
        app._handle_neutral_command(
            '/neutral auto scan example.com :: {"objective": "scan example.com"}'
        )
    )
    asyncio.run(app._handle_neutral_command("/neutral mcp find admin panels"))
    asyncio.run(app._handle_neutral_command("/neutral"))

    assert calls[0] == (
        "neutral_route",
        {
            "goal": "scan example.com",
            "scope": "all",
            "execute": True,
            "args": {"objective": "scan example.com"},
        },
    )
    assert calls[1][1]["goal"] == "find admin panels"
    assert calls[1][1]["scope"] == "mcp"
    assert calls[1][1]["execute"] is False
    assert "NEUTRAL ROUTER fake" in "\n".join(notes)
    assert "Neutral router" in "\n".join(notes)
