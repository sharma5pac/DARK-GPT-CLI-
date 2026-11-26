"""Skills are active by default; only an explicit opt-out deactivates them."""

import json

import pytest

from agent import skills as skill_mod


@pytest.fixture
def skill_dirs(tmp_path, monkeypatch):
    user = tmp_path / "skills"
    user.mkdir()
    state = tmp_path / "skills_state.json"
    for i in range(15):
        d = user / f"skill-{i:02d}"
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: skill-{i:02d}\ndescription: Test skill {i}\n---\nBody {i}\n",
            encoding="utf-8",
        )
    (user / "hack").mkdir()
    (user / "hack" / "SKILL.md").write_text(
        "---\nname: hack\ndescription: Master doctrine\n---\nMASTER BODY\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_mod, "USER_SKILLS_DIR", user)
    monkeypatch.setattr(skill_mod, "PROJECT_SKILLS_DIR", tmp_path / "project-skills")
    monkeypatch.setattr(skill_mod, "STATE_PATH", state)
    return user, state


def test_all_discovered_skills_active_without_state(skill_dirs):
    registry = skill_mod.SkillRegistry()
    assert registry.active_count() == len(registry.all()) == 16
    assert registry.is_active("hack")


def test_explicit_deactivation_survives_reload_and_activate_all_clears_it(skill_dirs):
    _, state = skill_dirs
    registry = skill_mod.SkillRegistry()
    registry.set_active("skill-03", False)
    data = json.loads(state.read_text())
    assert data["deactivated"] == ["skill-03"]
    reloaded = skill_mod.SkillRegistry()
    assert reloaded.active_count() == 15
    assert not reloaded.is_active("skill-03")
    assert reloaded.activate_all() == 16
    assert skill_mod.SkillRegistry().active_count() == 16


def test_large_active_library_uses_compact_catalog_with_master(skill_dirs):
    registry = skill_mod.SkillRegistry()
    text = registry.active_instructions()
    assert "ALL 16 DISCOVERED SKILLS ARE ACTIVE" in text
    assert "MASTER BODY" in text
    assert "skill-14: Test skill 14" in text
    assert "Body 14" not in text  # full documents are loaded via the `skills` tool
