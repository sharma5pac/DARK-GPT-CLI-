"""First-launch machine inventory tests."""

from __future__ import annotations

import asyncio
import time

from agent import tool_inventory as ti


def _fake_inventory() -> dict:
    return {
        "version": ti.VERSION,
        "scanned_at": time.time(),
        "binary_count": 3,
        "relevant_count": 2,
        "app_count": 1,
        "skill_count": 2,
        "folder_count": 2,
        "binaries": [],
        "relevant": [
            {"name": "aircrack-ng", "path": "/usr/bin/aircrack-ng", "relevant": True},
            {"name": "hashcat", "path": "/usr/bin/hashcat", "relevant": True},
        ],
        "apps": [{"name": "Burp Suite", "relevant": True, "gui": True}],
        "skills": [{"name": "hack"}, {"name": "report"}],
        "folders": [{"path": "/home/makavelli/Desktop", "entries": 12}],
    }


def test_inventory_cache_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(ti, "CACHE_PATH", tmp_path / "tool_inventory.json")
    monkeypatch.setattr(
        ti, "scan_binaries",
        lambda: [
            {"name": "aircrack-ng", "path": "/usr/bin/aircrack-ng", "relevant": True, "category": "password & encryption auditing", "tags": ["password"]},
            {"name": "hashcat", "path": "/usr/bin/hashcat", "relevant": True, "category": "password & encryption auditing", "tags": ["hash"]},
            {"name": "random", "path": "/usr/bin/random", "relevant": False, "category": "", "tags": []},
        ],
    )
    monkeypatch.setattr(ti, "scan_desktop_apps", lambda: [{"name": "Burp Suite", "relevant": True, "gui": True}])
    monkeypatch.setattr(ti, "scan_skills", lambda: [{"name": "hack"}, {"name": "report"}])
    monkeypatch.setattr(ti, "scan_folders", lambda: [{"path": "/home/makavelli/Desktop", "exists": True, "entries": 12}])

    data, scanned = ti.ensure_fresh()
    assert scanned is True
    assert ti.CACHE_PATH.exists()
    assert data["binary_count"] == 3
    assert data["relevant_count"] == 2

    data2, scanned2 = ti.ensure_fresh()
    assert scanned2 is False
    assert data2["relevant_count"] == 2

    summary = ti.context_summary(data2)
    assert "capability inventory" in summary.lower()
    assert "aircrack-ng" in summary and "hashcat" in summary
    assert "Burp Suite" in summary

    data3, scanned3 = ti.ensure_fresh(force=True)
    assert scanned3 is True and data3["app_count"] == 1
    print("PASS first-launch inventory cache and model summary")


def test_startup_inventory_worker_updates_panel(monkeypatch):
    from tui import app as app_module

    fake = _fake_inventory()
    monkeypatch.setattr(app_module.tool_inventory, "ensure_fresh", lambda force=False: (fake, True))
    summaries = []
    real_summary = app_module.tool_inventory.startup_summary

    def record_summary(data=None, scanned=False):
        text = real_summary(data, scanned)
        summaries.append((data, scanned, text))
        return text

    monkeypatch.setattr(app_module.tool_inventory, "startup_summary", record_summary)

    async def _run():
        app = app_module.DarkGPTApp()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause(); await pilot.pause()
            summaries.clear()
            await app._startup_inventory()
            await pilot.pause()
            panel = app.query_one("#plan-panel", app_module.PlanPanel)
            rendered = panel.content.plain
            assert "inventory ready" in rendered, rendered
            assert "[x]" in rendered, rendered
            assert summaries and summaries[-1][1] is True
            assert "FIRST-LAUNCH SCAN" in summaries[-1][2]

    asyncio.run(_run())
    print("PASS startup inventory worker updates live plan panel")
