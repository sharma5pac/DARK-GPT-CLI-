"""Markdown skill registry for the DarkGPT agent.

Skills are plain Markdown files with optional YAML-ish frontmatter::

    ---
    name: recon
    description: Safe reconnaissance workflow
    triggers: recon, scan, enumerate
    tools: shell, web_search
    ---
    Instructions the model should follow while the skill is active.

Search order: user skills first, then project skills. A project skill with the
same name overrides the user one. Active state is persisted so a toggle survives
a TUI restart.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

USER_SKILLS_DIR = Path.home() / ".config" / "darkgpt" / "skills"
PROJECT_SKILLS_DIR = Path.cwd() / ".darkgpt" / "skills"
STATE_PATH = Path.home() / ".config" / "darkgpt" / "skills_state.json"


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "skill"


def _split_list(value: str) -> List[str]:
    value = (value or "").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [part.strip().strip("'\"") for part in value.split(",") if part.strip()]


def parse_skill_text(text: str, path: Path) -> "Skill":
    name = path.stem
    description = ""
    triggers: List[str] = []
    tools: List[str] = []
    body = text

    if text.startswith("---"):
        lines = text.splitlines()
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is not None:
            meta_lines = lines[1:end]
            body = "\n".join(lines[end + 1 :]).strip()
            current_key = None
            for line in meta_lines:
                if not line.strip():
                    continue
                if ":" in line and not line.startswith((" ", "\t")):
                    key, value = line.split(":", 1)
                    current_key = key.strip().lower()
                    value = value.strip()
                    if current_key == "name" and value:
                        name = value
                    elif current_key == "description":
                        description = value
                    elif current_key == "triggers":
                        triggers = _split_list(value)
                    elif current_key == "tools":
                        tools = _split_list(value)
                elif current_key in ("triggers", "tools") and line.strip().startswith("-"):
                    item = line.strip().lstrip("-").strip().strip("'\"")
                    if current_key == "triggers":
                        triggers.append(item)
                    else:
                        tools.append(item)

    if not description:
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                description = line[:120]
                break
    return Skill(
        name=name.strip(),
        description=description.strip(),
        triggers=triggers,
        instructions=body.strip(),
        path=path,
        tools=tools,
        scope="project" if PROJECT_SKILLS_DIR in path.parents else "user",
    )


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    path: Path
    triggers: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    scope: str = "user"

    def render(self) -> str:
        lines = [f"## Skill: {self.name}"]
        if self.description:
            lines.append(self.description)
        if self.tools:
            lines.append(f"Preferred tools: {', '.join(self.tools)}")
        lines.append(self.instructions.strip())
        return "\n".join(line for line in lines if line).strip()


class SkillRegistry:
    def __init__(self, extra_dirs: Optional[Iterable[Path]] = None) -> None:
        self.dirs: List[Path] = [USER_SKILLS_DIR, PROJECT_SKILLS_DIR]
        if extra_dirs:
            self.dirs.extend(Path(p) for p in extra_dirs)
        self._skills: Dict[str, Skill] = {}
        self._active: set[str] = set()
        self.reload()

    # ---------------------------------------------------------------- loading
    def reload(self) -> None:
        found: Dict[str, Skill] = {}
        for directory in self.dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # Two layouts are supported:
            #   * flat files:   skills/my-skill.md
            #   * skill bundles: skills/my-skill/SKILL.md (+ references/, etc.)
            # Bundles keep their relative reference files intact, which is what
            # the REA skills need for their linked guides.
            paths = list(directory.glob("*.md")) + list(directory.glob("*/SKILL.md"))
            for path in sorted(paths):
                try:
                    skill = parse_skill_text(path.read_text(encoding="utf-8"), path)
                except Exception:
                    continue
                if skill.name:
                    found[skill.name] = skill  # later dirs (project) win
        self._skills = found
        self._active = {n for n in self._load_state() if n in found}

    def _load_state(self) -> List[str]:
        try:
            data = json.loads(STATE_PATH.read_text())
            return [str(x) for x in data.get("active", [])]
        except Exception:
            return []

    def _save_state(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({"active": sorted(self._active)}, indent=2))

    # ------------------------------------------------------------------- query
    def all(self) -> List[Skill]:
        return [self._skills[k] for k in sorted(self._skills)]

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def is_active(self, name: str) -> bool:
        return name in self._active

    def active(self) -> List[Skill]:
        return [self._skills[n] for n in sorted(self._active) if n in self._skills]

    def active_count(self) -> int:
        return len(self.active())

    def active_instructions(self) -> str:
        skills = self.active()
        if not skills:
            return ""
        return "\n\n".join(s.render() for s in skills)

    # ------------------------------------------------------------------ mutate
    def toggle(self, name: str) -> bool:
        if name not in self._skills:
            return False
        if name in self._active:
            self._active.discard(name)
            state = False
        else:
            self._active.add(name)
            state = True
        self._save_state()
        return state

    def set_active(self, name: str, active: bool) -> None:
        if name not in self._skills:
            return
        if active:
            self._active.add(name)
        else:
            self._active.discard(name)
        self._save_state()

    # ------------------------------------------------------------- file writes
    def _target_path(self, name: str, scope: str = "user") -> Path:
        directory = PROJECT_SKILLS_DIR if scope == "project" else USER_SKILLS_DIR
        return directory / f"{_slug(name)}.md"

    def _render_file(self, name: str, description: str, triggers: List[str], tools: List[str], instructions: str) -> str:
        header = [
            "---",
            f"name: {name.strip()}",
            f"description: {description.strip()}",
            f"triggers: {', '.join(triggers)}",
            f"tools: {', '.join(tools)}",
            "---",
            "",
        ]
        return "\n".join(header) + instructions.strip() + "\n"

    def create(
        self,
        name: str,
        description: str = "",
        triggers: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        instructions: str = "",
        scope: str = "user",
    ) -> Skill:
        path = self._target_path(name, scope)
        if path.exists():
            raise FileExistsError(f"skill already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self._render_file(name, description, triggers or [], tools or [], instructions),
            encoding="utf-8",
        )
        self.reload()
        skill = self._skills.get(name)
        if skill is None:
            raise RuntimeError(f"skill was written but could not be parsed: {path}")
        return skill

    def update(
        self,
        name: str,
        description: Optional[str] = None,
        triggers: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        instructions: Optional[str] = None,
    ) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(name)
        path = skill.path
        path.write_text(
            self._render_file(
                name,
                description if description is not None else skill.description,
                triggers if triggers is not None else skill.triggers,
                tools if tools is not None else skill.tools,
                instructions if instructions is not None else skill.instructions,
            ),
            encoding="utf-8",
        )
        self.reload()
        updated = self._skills.get(name)
        if updated is None:
            raise RuntimeError(f"skill was written but could not be parsed: {path}")
        return updated

    def rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "path": str(s.path),
                "scope": s.scope,
                "active": s.name in self._active,
                "triggers": s.triggers,
                "tools": s.tools,
            }
            for s in self.all()
        ]
