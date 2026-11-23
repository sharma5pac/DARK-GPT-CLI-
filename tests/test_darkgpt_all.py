"""Consolidated DarkGPT verification suite (rebuilt after temp cleanup)."""
import asyncio, json, os, sys, tempfile
from pathlib import Path

ROOT = "/home/makavelli/Desktop/darkgpt/DarkGPT"
sys.path.insert(0, ROOT); os.chdir(ROOT)

from mcp_manager.client import MCPManager
async def _noop_start(self, config): return None
MCPManager.start_from_config = _noop_start

import tui.app as A
def fake_catalog(force_refresh=False, provider=None):
    model_id = "openrouter/model-a" if provider != "nvidia" else "nvidia/model-b"
    return {"free": [{"id": model_id, "name": model_id, "context": 128000}], "uncensored": []}
A.get_catalog = fake_catalog
A.discover_local_models = lambda: []

from agent import cline_tools, core as C, browser as B, skills as skills_util, session as session_util
from utils import appconfig, permissions
from utils import get_response as G
from utils.get_response import NVIDIA_NON_CHAT_MARKERS

PASS = []
def ok(label): PASS.append(label); print("  PASS", label)


# ---------------------------------------------------------------- non-TUI checks
def check_network_resilience():
    """A transient resolver stall must be retried, not reported as fatal DNS."""
    import requests as rq
    original_post, original_sleep = G.requests.post, G.time.sleep
    calls = []
    G.time.sleep = lambda _s: None
    try:
        def flaky_post(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                raise rq.ConnectionError(
                    "HTTPSConnectionPool(host='integrate.api.nvidia.com', port=443): "
                    "Max retries exceeded (Caused by NameResolutionError: "
                    "[Errno -3] Temporary failure in name resolution)"
                )
            class Resp:
                status_code = 200
                def json(self): return {"choices": [{"message": {"content": "ok"}}]}
            return Resp()
        G.requests.post = flaky_post
        resp = G.post_with_retry("https://integrate.api.nvidia.com/v1/chat/completions",
                                 headers={}, payload={}, timeout=5)
        assert len(calls) == 2 and resp.status_code == 200, calls
        ok("transient DNS error is retried, not fatal")

        # A DNS-flavoured timeout must be described as a timeout.
        msg = C.clean_network_error(rq.ConnectionError("DNS lookup for x timed out after 5s"))
        assert "timed out" in msg and "DNS lookup failed" not in msg, msg
        ok("DNS timeout not mislabelled as permanent failure")
    finally:
        G.requests.post, G.time.sleep = original_post, original_sleep


def check_config_and_providers():
    cfg = appconfig.load_config()
    assert cfg["provider"] in ("openrouter", "nvidia"), cfg["provider"]
    assert cfg["model"] and "/" in cfg["model"], cfg["model"]
    assert cfg.get("mode") in ("Build", "Ask")
    ok("config: provider/model/mode valid")
    assert str(cfg.get("model_nvidia", "")).startswith("nvidia/"), cfg.get("model_nvidia")
    assert NVIDIA_NON_CHAT_MARKERS, "NVIDIA non-chat filter missing"
    ok("provider catalog filter present")


def check_self_learning():
    from agent import memory as M

    with tempfile.TemporaryDirectory() as d:
        for _ in range(2):
            M.remember(d, "failed-approach", "fetch_web_content: only http:// and https:// URLs are supported")
        entries = M.by_kind(d, {"failed-approach"}, k=5)
        assert len(entries) == 1 and entries[0]["count"] == 2, entries
        block = M.format_lessons(entries)
        assert "hit 2x" in block and "fetch_web_content" in block, block
        ok("lessons dedupe + strengthen")

        M.remember(d, "pattern", "kala nmap returns empty error twice; switch to shell nmap")
        assert "switch to shell" in M.format_patterns(M.by_kind(d, {"pattern"}, k=5))
        ok("pattern capture")

        extras = C.context_util.system_extras(d, "fetch a url", git=False, memory=False, self_learning=True)
        assert "Learned lessons" in extras and "Observed patterns" in extras, extras
        ok("lessons/patterns injected into system extras")

        removed = M.forget_kinds(d, {"failed-approach"})
        assert removed == 1 and not M.by_kind(d, {"failed-approach"}, k=5)
        ok("forget lessons")

    assert C._short_error('{"error": "bad argument shape"}') == "bad argument shape"
    assert C._is_transient("connection timed out after 5s") is True
    assert C._is_transient("invalid tool argument: expected array") is False
    assert C._error_signature("HTTP 500 at /a") == C._error_signature("HTTP 502 at /b")
    ok("failure classifier (transient vs durable)")


def check_fetch_normalisation():
    from agent import cline_tools as CT
    raw = {"requests": '[{"url": "https://example.com/", "prompt": "title"}]'}
    assert CT.normalise_web_requests(raw) == [{"url": "https://example.com/", "prompt": "title"}]
    assert CT.normalise_web_requests('[{"url":"https://a.test"}]') == [{"url": "https://a.test", "prompt": ""}]
    assert CT.normalise_web_requests("https://plain.test") == [{"url": "https://plain.test", "prompt": ""}]
    ok("fetch_web_content decodes stringified JSON")


def check_cline_tools():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.txt"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        r = cline_tools.read_files(str(p), d)
        assert "alpha" in json.dumps(r, ensure_ascii=False), r
        ok("cline read_files")
        s = cline_tools.search_codebase("beta", d)
        assert "beta" in json.dumps(s, ensure_ascii=False), s
        ok("cline search_codebase")
        c = cline_tools.run_commands("echo cline-ok", d)
        assert "cline-ok" in json.dumps(c, ensure_ascii=False), c
        ok("cline run_commands")
        e = cline_tools.editor(str(p), "BETA", old_text="beta", cwd=d)
        assert "BETA" in p.read_text(), e
        ok("cline editor")
    assert permissions.risk_of("browser", {}) == "safe"
    ok("permissions: browser safe")


def check_session_and_skills():
    entry = session_util.create(os.getcwd(), [
        {"role": "user", "content": "suite q"},
        {"role": "assistant", "content": "suite a"},
    ], model="test-model", label="suite", last_prompt="suite q")
    assert entry and entry.get("id"), entry
    loaded = session_util.load(str(entry["id"]))
    assert loaded and (loaded.get("turns") or loaded.get("messages")), loaded
    ok("session create/load")
    assert "suite q" in session_util.summary_text(loaded)
    assert "suite q" in session_util.summary_text(entry)  # hardened against metadata-only shape
    ok("session summary (full entry + metadata shapes)")
    reg = skills_util.SkillRegistry()
    assert len(reg.all()) > 0, "no skills loaded"
    ok(f"skills registry ({len(reg.all())} bundles)")


def check_browser():
    result = B.run([
        {"action": "navigate", "url": "data:text/html,<h1 id='x'>ALL-SUITE-OK</h1>"},
        {"action": "extract", "selector": "#x", "mode": "text"},
    ], session="allsuite", engine="auto", timeout=60)
    assert result.get("ok"), result
    assert result.get("engine") in ("playwright", "selenium"), result
    assert "ALL-SUITE-OK" in str(result["steps"][1].get("result")), result
    ok(f"browser tool real run ({result['engine']})")


# ---------------------------------------------------------------- TUI checks
COPIED = []
class FakeResult:
    ok = True; backend = "test"; detail = ""
def fake_copy(text, app=None, also_primary=True):
    COPIED.append(text); return FakeResult()
A.clipboard.copy_text = fake_copy
PASTED = []
A.clipboard.paste_text = lambda app=None: (PASTED.append(1) or "p1\np2\np3\np4\np5\np6")
NOTES = []
A.DarkGPTApp._system_note = lambda self, text: NOTES.append(str(text))


async def check_tui():
    app = A.DarkGPTApp()
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.pause(); await pilot.pause()
        area = app.query_one("#prompt-input", A.PromptArea)
        mode_btn = app.query_one("#mode-btn", A.Button)
        hint = app.query_one("#hint-line", A.Static)

        # Mode toggle from prompt and from other focus.
        assert str(app.mode) == "Build" and "Ctrl+Alt+C" in str(hint.content)
        ok("hint line visible with copy/paste keys")
        startup = " ".join(NOTES)
        for token in ("WEB:", "built-in browser", "/pending", "/provider", "/status", "Tab Build/Ask", "LEARN:", "hack skill active"):
            assert token in startup, (token, startup[:400])
        ok("startup note highlights web/queue/provider features")
        await pilot.press("tab"); await pilot.pause()
        assert str(app.mode) == "Ask" and "mode-ask" in mode_btn.classes, (app.mode, mode_btn.classes)
        await pilot.click("#conversation"); await pilot.pause()
        await pilot.press("tab"); await pilot.pause()
        assert str(app.mode) == "Build" and "mode-build" in mode_btn.classes
        ok("Tab toggles mode from any focus + styled badge")

        # Modal Tab guard.
        await pilot.press("ctrl+g"); await pilot.pause()
        assert isinstance(app.screen, A.SkillsManagerScreen)
        before = str(app.mode)
        await pilot.press("tab"); await pilot.pause()
        assert str(app.mode) == before
        await pilot.press("escape"); await pilot.pause()
        ok("modal Tab guard")

        # Copy/paste terminal-safe keys.
        area.load_text("SELECT-ME"); area.focus()
        area.selection = ((0, 0), (0, 9)); await pilot.pause()
        COPIED.clear(); await pilot.press("ctrl+alt+c"); await pilot.pause()
        assert COPIED and "SELECT-ME" in COPIED[-1], COPIED
        await pilot.press("alt+c"); await pilot.pause()
        assert len(COPIED) == 2, COPIED
        ok("copy via Ctrl+Alt+C / Alt+C")
        area.load_text(""); area._full_text = ""
        await pilot.press("ctrl+alt+v"); await pilot.pause()
        assert "p1" in area.text and "p6" in area.text and "[Pasted" not in area.text, repr(area.text)
        ok("paste verbatim multi-line")

        # Model picker opens and has grouped rows.
        await pilot.press("ctrl+l"); await pilot.pause()
        assert isinstance(app.screen, A.ModelPickerScreen), type(app.screen)
        picker = app.screen.query_one("#model-list", A.OptionList)
        for _ in range(20):
            await pilot.pause()
            if picker.options:
                break
        ids = [str(o.id or "") for o in picker.options]
        assert any(i.startswith("openrouter|") for i in ids), ids
        assert any(i.startswith("nvidia|") for i in ids), ids
        prompts = [str(o.prompt) for o in picker.options]
        assert any("──" in p and "OpenRouter" in p for p in prompts), prompts
        assert any("──" in p and "NVIDIA" in p for p in prompts), prompts
        await pilot.press("down"); await pilot.pause()
        assert picker.highlighted is not None
        await pilot.press("escape"); await pilot.pause()
        ok("model picker groups both providers + arrow navigation")

        # Badge modals.
        for key, cls in (("ctrl+m", A.MCPManagerScreen), ("ctrl+t", A.ToolsManagerScreen)):
            await pilot.press(key); await pilot.pause()
            assert isinstance(app.screen, cls), (key, type(app.screen))
            await pilot.press("escape"); await pilot.pause()
        ok("MCP/Tools bindings open modals")

        # Copy-all + slash command.
        app.messages = [{"role": "user", "content": "hello suite"}, {"role": "assistant", "content": "answer suite"}]
        conv = app.query_one("#conversation", A.VerticalScroll)
        await conv.mount(A.ChatMessage("user", "hello suite"))
        await conv.mount(A.ChatMessage("assistant", "answer suite"))
        await pilot.pause()
        COPIED.clear(); app.action_copy_all(); await pilot.pause()
        assert COPIED and "hello suite" in COPIED[-1], COPIED
        ok("copy-all")

        # Pending queue while busy.
        calls = []
        class DummyWorker:
            def cancel(self): pass
        def fake_run(self, text, echo_user=True):
            calls.append(text); self.busy = False; return DummyWorker()
        original_run = app.run_prompt
        app.run_prompt = fake_run.__get__(app, A.DarkGPTApp)
        app.busy = True
        area.load_text("queued-1"); area._full_text = "queued-1"; await app.submit_prompt()
        area.load_text("queued-2"); area._full_text = "queued-2"; await app.submit_prompt()
        assert app._pending == ["queued-1", "queued-2"], app._pending
        await app.execute_command("/pending clear")
        assert app._pending == []
        ok("pending queue + /pending clear")
        app.run_prompt = original_run

        # Self-learning commands.
        from agent import memory as M
        NOTES.clear()
        await app.execute_command("/learn off")
        assert app.self_learning_enabled is False
        await app.execute_command("/learn on")
        assert app.self_learning_enabled is True
        ok("/learn toggles self-learning")

        cwd = os.getcwd()
        mem_path = M.path_for(cwd)
        mem_backup = mem_path.read_text() if mem_path.exists() else None
        M.remember(cwd, "failed-approach", "suite lesson: bad argument shape")
        M.remember(cwd, "pattern", "suite pattern: repeated signature")
        NOTES.clear()
        await app.execute_command("/lessons")
        assert any("suite lesson" in n for n in NOTES), NOTES
        NOTES.clear()
        await app.execute_command("/patterns")
        assert any("suite pattern" in n for n in NOTES), NOTES
        ok("/lessons and /patterns show learned entries")
        await app.execute_command("/lessons clear")
        await app.execute_command("/patterns clear")
        assert not M.by_kind(cwd, {"failed-approach", "pattern"}, k=5)
        ok("/lessons clear and /patterns clear")
        if mem_backup is not None:
            mem_path.write_text(mem_backup)
        elif mem_path.exists():
            mem_path.unlink()

        # Repeat guard helper exists and is result-aware.
        assert (C.MAX_REPEAT_CALLS, C.MAX_TOOLS_PER_TURN, C.MAX_STEPS) == (3, 14, 14)
        assert (C.MAX_VERIFY_ROUNDS, C.MAX_REVIEW_ROUNDS, C.MAX_SUBAGENTS_PER_TURN) == (2, 1, 3)
        ok("guardrails configured (repeat/tools/steps/verify/review/subagents)")


async def main():
    check_network_resilience()
    check_self_learning()
    check_fetch_normalisation()
    check_config_and_providers()
    check_cline_tools()
    check_session_and_skills()
    check_browser()
    await check_tui()
    print(f"\nALL DARKGPT CHECKS PASSED ({len(PASS)} sections)")

asyncio.run(main())
