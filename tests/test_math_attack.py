"""Math-for-attacks runtime: bounded computation, evidence gates, learning."""

import asyncio

import pytest

from agent import evidence, knowledge_graph as kg, math_attack as math_mod
from agent import reporting, semantic_memory as semantic_util
from agent.core import ToolRegistry


@pytest.fixture
def isolated_math(tmp_path, monkeypatch):
    monkeypatch.setattr(math_mod, "STATE_PATH", tmp_path / "math_state.json")
    monkeypatch.setattr(math_mod, "WORKER_PATH", math_mod.WORKER_PATH)
    monkeypatch.setattr(kg, "ROOT", tmp_path / "targets")
    monkeypatch.setattr(semantic_util, "ROOT", tmp_path / "memory")
    monkeypatch.setattr(semantic_util, "STORE", tmp_path / "memory" / "semantic.json")
    return tmp_path


def test_analyze_routes_prng_and_writes_evidence(isolated_math):
    result = math_mod.run(
        {
            "action": "analyze",
            "goal": "check session token entropy",
            "domain": "prng",
            "input": {"samples": [0, 1, 1, 0, 1, 0, 1, 1], "test": "entropy"},
        },
        target="math-test",
    )
    assert result["ok"] is True
    assert result["domain"] == "prng"
    assert result["entropy_bits"] > 0
    assert result["evidence_ref"]
    assert evidence.has_ref("math-test", result["evidence_ref"])
    assert result["mastery"] > 0


def test_chi_square_buckets_are_deterministic(isolated_math):
    payload = {"samples": ["a", "b", "c", "a", "b", "a"], "test": "chi_square", "buckets": 8}
    first = math_mod.compute("stats", payload, domain="stats")
    second = math_mod.compute("stats", payload, domain="stats")
    assert first["ok"] is True
    assert first["chi_square"] == second["chi_square"]


def test_verify_residual_gate(isolated_math):
    good = math_mod.run({"action": "verify", "expr": "2+2", "expected": "4"})
    bad = math_mod.run({"action": "verify", "expr": "2+2", "expected": "5"})
    assert good["ok"] is True and good["verified"] is True
    assert bad["ok"] is True and bad["verified"] is False


def test_learn_gate_and_recall_persist(isolated_math):
    candidate = math_mod.learn("weak nonces", "low entropy tokens are guessable", domain="prng", target="math-test")
    assert candidate["status"] == "candidate"
    entry = evidence.record("math-test", tool="manual", result="proof artifact")
    verified = math_mod.learn(
        "modular inverse",
        "gcd(a,m)=1 is required before using a modular inverse",
        domain="crypto",
        evidence_ref=entry["id"],
        target="math-test",
    )
    assert verified["status"] == "verified"
    assert verified["lesson"]["evidence_ref"] == entry["id"]

    recalled = math_mod.recall("modular inverse", limit=10)
    assert recalled["count"] >= 1
    assert any(item["title"] == "modular inverse" for item in recalled["lessons"])
    status = math_mod.status()
    assert status["lessons"] == 2
    assert status["verified_lessons"] == 1


def test_rh_status_keeps_crypto_guardrail(isolated_math):
    payload = math_mod.rh_status()
    text = payload["text"].lower()
    assert "formalize -> compute -> verify -> crypto impact" in text
    assert "does not break rsa" in text
    assert "prize status pending" in text
    assert "grh for dirichlet l-functions" in text


def test_report_includes_math_analysis(isolated_math):
    data = reporting.build_report_data("math-test")
    assert "math_analysis" in data
    assert "does not break rsa" in data["math_analysis"]["rh_status"]["crypto_impact"].lower()
    assert "Math analysis and learned rules" in reporting.build_markdown("math-test")


def test_tool_spec_includes_math_attack_and_tui_command():
    names = {s["function"]["name"] for s in ToolRegistry().local_specs()}
    assert "math_attack" in names
    from tui.app import COMMANDS

    assert any(command == "/math" for command, _ in COMMANDS)


def test_tui_math_command_renders_cards(isolated_math, monkeypatch):
    from tui import app as app_mod

    notes = []
    monkeypatch.setattr(app_mod.DarkGPTApp, "_system_note", lambda self, text: notes.append(str(text)))

    async def run():
        app = app_mod.DarkGPTApp()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            await app.execute_command("/math domains")
            await app.execute_command("/math rh")
            await pilot.pause()

    asyncio.run(run())
    joined = "\n".join(notes)
    assert "FORMALIZE -> COMPUTE -> VERIFY -> CRYPTO IMPACT" in joined
    assert "Riemann hypothesis status" in joined
