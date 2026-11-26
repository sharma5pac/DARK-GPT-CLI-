"""Focused coverage for the ARTEX-style goals/worker/retester loop.

All state is isolated to tmp_path: no live config, evidence or targets are read.
"""

import asyncio
import json

import pytest

from agent import knowledge_graph as kg
from utils import appconfig


@pytest.fixture(autouse=True)
def isolated_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "ROOT", tmp_path / "targets")
    cfg = tmp_path / "config"
    monkeypatch.setattr(appconfig, "CONFIG_DIR", cfg)
    monkeypatch.setattr(appconfig, "CONFIG_PATH", cfg / "config.json")
    cfg.mkdir(parents=True, exist_ok=True)
    appconfig.set_value("active_target", "")
    yield


def _seed_assets(target="artex.test"):
    from agent import knowledge_graph as k
    k.add_host(target, "example.test", {}, source="test")
    k.add_endpoint(target, "https://example.test/app", {}, source="test")
    return target


def test_goal_crud_persistence_and_dedupe():
    from agent import engagement as eng

    t = "goals.test"
    first = eng.add_goal(t, "Verify exposed admin panel", scope=["example.test"],
                         success_criteria=["evidence-backed asset"], source="test")
    assert first["ok"] and first["goal"]["status"] == "open"
    again = eng.add_goal(t, "verify exposed admin panel", scope=["example.test"],
                         success_criteria=["different wording"], source="test")
    assert again["deduplicated"] is True
    data = eng.load(t)
    assert len(data["goals"]) == 1
    assert "different wording" in data["goals"][0]["success_criteria"]
    assert "Verify exposed admin panel" in json.dumps(eng.load(t))


def test_scope_breaks_into_verifiable_goals():
    from agent import engagement as eng

    result = eng.set_scope("scope.test", ["example.test", "https://example.test/app"])
    assert result["count"] == 2
    goals = eng.load("scope.test")["goals"]
    assert all(g["success_criteria"] for g in goals)
    assert all(g["scope"] for g in goals)


def test_work_graph_anchors_and_coverage_are_derived():
    from agent import engagement as eng, work_graph as wg

    t = _seed_assets("coverage.test")
    eng.set_scope(t, ["example.test", "https://example.test/app"])
    emitted = eng.emit_intents(t, limit=4)
    assert emitted["emitted"] == 2
    assert wg.coverage(t)["percent"] == 0.0
    work = wg.load(t)
    for node in work["nodes"].values():
        if node.get("kind") == "intent" and node.get("anchors"):
            wg.set_status(t, node["key"], "done", note="test proof")
    cov = wg.coverage(t)
    assert cov["percent"] == 100.0
    assert cov["covered"] == 2
    assert any(a.get("node_key") for n in wg.load(t)["nodes"].values()
               for a in n.get("anchors", []))


def test_intent_lifecycle_bounded_attempts_and_action_validation():
    from agent import engagement as eng

    t = _seed_assets("intents.test")
    eng.set_scope(t, ["example.test"])
    emitted = eng.emit_intents(t, limit=2)
    assert emitted["emitted"] >= 1
    iid = emitted["intents"][0]["id"]
    claimed = eng.claim_intent(t, "unit-worker", iid)
    assert claimed and claimed["status"] == "claimed" and claimed["attempts"] == 1
    assert eng.claim_intent(t, "other-worker", iid) is None
    done = eng.complete_intent(t, iid, ok=True, evidence_ref="ev-1", summary="ok")
    assert done["status"] == "done"
    assert eng.claim_intent(t, "again", iid) is None

    actions = eng.validate_actions([
        {"tool": "script", "args": {"code": "print(1)"}},
        {"tool": "engagement_run", "args": {}},
        {"tool": "drop_all", "args": {}},
        {"tool": "shell", "args": {"command": "true"}},
    ])
    assert [a["tool"] for a in actions] == ["script", "shell"]
    assert eng.MAX_INTENT_ATTEMPTS == 3 and eng.MAX_ACTIONS_PER_INTENT == 4


def test_validation_intent_without_proof_is_not_done():
    from agent import engagement as eng

    t = _seed_assets("proof.test")
    eng.set_scope(t, ["example.test"])
    data = eng.load(t)
    data["intents"].append({
        "id": "validate-1", "goal_id": "", "kind": "validate",
        "text": "validate", "anchor": {}, "success_signal": "evidence",
        "budget": {"tools": 1, "steps": 1}, "attempts": 1, "status": "running",
        "owner": "worker", "created_at": 0, "updated_at": 0, "actions": [],
    })
    eng.save(t, data)
    result = eng.complete_intent(t, "validate-1", ok=True, summary="looks vulnerable")
    assert result["status"] == "failed"
    assert "proof gate" in result["intent"]["summary"]


def test_context_block_exposes_live_goals():
    from agent import engagement as eng

    t = _seed_assets("context.test")
    eng.set_scope(t, ["example.test"])
    block = eng.context_block(t)
    assert "Engagement" in block and "goal" in block and "engagement_run" in block


def test_core_dispatch_engagement_and_worker(tmp_path):
    from agent import engagement as eng
    from agent.core import ToolRegistry

    t = _seed_assets("dispatch.test")
    eng.set_scope(t, ["example.test"])
    eng.emit_intents(t, limit=2)
    appconfig.set_value("active_target", t)
    registry = ToolRegistry(permission_mode="full", hooks_enabled=False)

    events = []
    payload = json.loads(asyncio.run(registry._dispatch(
        "engagement", {"action": "summary"}, on_event=events.append)))
    assert payload["ok"] is True
    assert any(e.get("type") == "engagement" for e in events)

    events.clear()
    payload = json.loads(asyncio.run(registry._dispatch(
        "engagement", {"action": "intents", "limit": 2}, on_event=events.append)))
    assert payload["emitted"] >= 1

    payload = json.loads(asyncio.run(registry._dispatch(
        "engagement_run", {"actions": [{"tool": "attack_paths", "args": {"max_paths": 3}}]},
        on_event=events.append)))
    assert payload["ok"] is True and payload["status"] == "done"
    assert any(e.get("type") == "engagement_intent" and e.get("status") == "running"
               for e in events)
    assert any(e.get("type") == "engagement_intent" and e.get("status") == "done"
               for e in events)
    assert eng.load(t)["workers"]
    appconfig.set_value("active_target", "")


def test_retest_updates_stable_node_and_records_evidence():
    from agent import evidence as ev, engagement as eng, reporting
    from agent.core import ToolRegistry

    t = "retest.test"
    node = kg.add_finding(t, "SQL injection in q", "high", "details", "", {}, source="test")
    key = node["key"]
    kg.update_node_attrs(t, key, {"repro_command": "true"}, source="test")
    before = len(kg.query(t, kind="finding"))
    appconfig.set_value("active_target", t)
    registry = ToolRegistry(permission_mode="full", hooks_enabled=False)
    payload = json.loads(asyncio.run(registry._dispatch(
        "verify_findings", {"action": "retest", "timeout": 10})))
    assert payload["ok"] and payload["results"][0]["verdict"] == "still_reproducible"
    after = kg.query(t, kind="finding")
    assert len(after) == before  # no duplicate node by title
    updated = next(n for n in after if n["key"] == key)
    assert updated["attrs"]["status"] == "proved"
    assert updated["attrs"]["evidence_ref"] == payload["results"][0]["evidence_ref"]
    assert eng.load(t)["retests"]
    assert ev.summary(t)["count"] >= 1
    assert "Pentest Report" in reporting.build_markdown(t)
    appconfig.set_value("active_target", "")


def test_poc_fields_and_retest_history_render_in_all_formats():
    from agent import engagement as eng, reporting

    t = "report.test"
    node = kg.add_finding(t, "Open redirect", "medium", "details", "", {}, source="test")
    kg.update_node_attrs(t, node["key"], {
        "poc_steps": ["Visit /redirect?to=evil", "Observe 302"],
        "poc_command": "curl -sI https://example.test/redirect?to=evil",
        "poc_script": "/tmp/poc.py", "poc_sha256": "abc123",
    }, source="test")
    eng.record_retest(t, "Open redirect", "still_reproducible", evidence_ref="ev-9")
    data = reporting.build_report_data(t)
    finding = data["findings"][0]
    assert finding["poc_steps"][0].startswith("Visit")
    assert finding["last_retest_verdict"] == "still_reproducible"
    assert data["engagement"]["retests"]
    assert "PoC command" in reporting.build_markdown(t)
    assert "PoC steps" in reporting.build_html(t)
    csv_text = reporting.build_csv(t)
    assert "poc_sha256" in csv_text and "last_retest_verdict" in csv_text
    assert "still_reproducible" in json.dumps(json.loads(reporting.build_json(t)))


def test_worker_roles_have_schemas():
    from agent import subagents

    for role in ("goal_decomposer", "intent_planner", "coverage_auditor",
                 "poc_builder", "retest_reviewer"):
        assert role in subagents.ROLE_SYSTEM
        assert role in subagents.ROLE_SCHEMAS
        assert "required" in subagents.ROLE_SCHEMAS[role]
