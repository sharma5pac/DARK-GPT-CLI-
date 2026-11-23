"""Project rules file loader (CLAUDE.md-style) with auto-detected conventions.

Rules are plain Markdown. DarkGPT reads them into the system prompt on every
turn, so conventions survive context compaction and session restarts.

Lookup order (all files found are merged, highest priority first):

1. <repo>/.darkgpt/rules.md     - managed by /rules add
2. <repo>/DARKGPT.md
3. <repo>/CLAUDE.md             - familiar name, fully supported
4. <repo>/AGENTS.md
5. ~/.config/darkgpt/rules.md   - global personal rules

Search walks from the current directory up to the git root (or filesystem
root), so running from a subdirectory still picks up repository rules.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

MAX_RULES_CHARS = 14_000
NAMES = (".darkgpt/rules.md", "DARKGPT.md", "CLAUDE.md", "AGENTS.md")


def repo_root(cwd: str = "") -> Path:
    try:
        from utils import vcs

        root = vcs.root(cwd or os.getcwd())
        if root:
            return Path(root)
    except Exception:
        pass
    return Path(cwd or os.getcwd()).resolve()


def candidate_paths(cwd: str = "") -> List[Path]:
    start = Path(cwd or os.getcwd()).resolve()
    root = repo_root(str(start))
    parts: List[Path] = []
    current = start
    while True:
        parts.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent
        if current == Path.home().parent:
            break
    seen = set()
    paths: List[Path] = []
    for base in parts:
        for name in NAMES:
            p = base / name
            if p.is_file() and p not in seen:
                seen.add(p)
                paths.append(p)
    global_rules = Path.home() / ".config" / "darkgpt" / "rules.md"
    if global_rules.is_file() and global_rules not in seen:
        paths.append(global_rules)
    return paths


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace").strip()
    except Exception:
        return ""


def load(cwd: str = "") -> List[Dict[str, str]]:
    """Return ``[{path, text}]`` for every rules file found."""
    return [{"path": str(p), "text": text} for p in candidate_paths(cwd) if (text := _read(p))]


def format_rules(cwd: str = "") -> str:
    entries = load(cwd)
    if not entries:
        return ""
    blocks: List[str] = []
    used = 0
    for entry in entries:
        label = os.path.relpath(entry["path"], cwd or os.getcwd())
        text = entry["text"]
        remaining = MAX_RULES_CHARS - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n[...rules truncated...]"
        blocks.append(f"## Project rules: {label}\n{text}")
        used += len(text)
    return "# Project rules (follow these)\n\n" + "\n\n".join(blocks)


def add_rule(text: str, cwd: str = "") -> Path:
    """Append a rule to the repo-managed rules file (never overwrites)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty rule")
    path = repo_root(cwd) / ".darkgpt" / "rules.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read(path)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    addition = f"\n- {text}  _{stamp}_"
    path.write_text((existing + addition).strip() + "\n")
    return path


def show(cwd: str = "") -> str:
    entries = load(cwd)
    if not entries:
        return "No project rules found. Add one with /rules add <text>."
    lines: List[str] = []
    for entry in entries:
        lines.append(f"--- {entry['path']} ---\n{entry['text'][:4000]}")
    return "\n\n".join(lines)[:MAX_RULES_CHARS]


def detected_facts(cwd: str = "") -> str:
    """Conventions derived from the repo itself, always current."""
    try:
        from utils import vcs, verify

        facts: List[str] = []
        commands = verify.detect(cwd or os.getcwd())
        if commands:
            facts.append("Detected checks: " + "; ".join(commands[:6]))
        root = vcs.root(cwd or os.getcwd())
        if root:
            facts.append(f"Git root: {root}")
        base = Path(root or cwd or os.getcwd())
        markers = {
            "package.json": "Node/npm", "pyproject.toml": "Python (pyproject)",
            "requirements.txt": "Python (pip)", "go.mod": "Go modules",
            "Cargo.toml": "Rust/Cargo", "Makefile": "Make",
        }
        found = [label for name, label in markers.items() if (base / name).exists()]
        if found:
            facts.append("Project type: " + ", ".join(found))
        return "\n".join(facts)
    except Exception:
        return ""
