"""Browser worker for DarkGPT.

Runs in *system* Python (where Selenium / Playwright live), not in DarkGPT's
venv. Reads one JSON job from argv[1] and prints one JSON result. DarkGPT calls
this through the ``browser`` tool, so the model never has to write scripts by
hand.

Supported actions (each step is a dict with an ``action`` key):

    navigate  url
    fill      selector, value                (fallback: name/id/label text)
    click     selector | text | xpath
    extract   selector (default body), mode=text|html|attr|links|title, attr, all
    screenshot path
    scroll    direction=top|bottom | pixels
    wait      selector | ms
    back
    cookies                                  (return current cookies)
    js        script                         (evaluate JavaScript)

An ``auto`` engine tries Playwright first and falls back to Selenium on any
launch/importer failure, so a broken Playwright install never blocks the task.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List


def _select(element: Any, selector: str = "", text: str = "", xpath: str = "") -> Any:
    """Find an element by CSS, visible text, or XPath, in that order."""
    if xpath:
        return element.find_element("xpath", xpath)
    if selector:
        try:
            return element.find_element("css selector", selector)
        except Exception:
            pass
        for attr in ("id", "name", "placeholder", "aria-label"):
            try:
                return element.find_element("xpath", f".//*[@{attr}='{selector}']")
            except Exception:
                continue
        try:
            return element.find_element("xpath", f".//*[normalize-space(text())='{selector}']")
        except Exception as exc:
            raise RuntimeError(f"element not found: {selector}") from exc
    if text:
        for tag in ("a", "button", "input", "span", "div", "li"):
            try:
                return element.find_element("xpath", f".//{tag}[normalize-space(text())='{text}']")
            except Exception:
                continue
        try:
            return element.find_element("xpath", f".//*[contains(text(),'{text}')]")
        except Exception as exc:
            raise RuntimeError(f"text not found: {text}") from exc
    raise RuntimeError("no selector, text or xpath supplied")


def _apply_cookies(driver: Any, cookies: List[Dict[str, Any]]) -> None:
    for cookie in cookies or []:
        try:
            driver.add_cookie(dict(cookie))
        except Exception:
            pass


def _cookie_dicts(driver: Any) -> List[Dict[str, Any]]:
    try:
        return [dict(c) for c in driver.get_cookies()]
    except Exception:
        return []


def _run_selenium(job: Dict[str, Any], steps: List[Dict[str, Any]], out: Dict[str, Any]) -> None:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    import shutil

    options = Options()
    options.binary_location = shutil.which("chromium") or shutil.which("google-chrome") or "/usr/bin/chromium"
    if job.get("headless", True):
        options.add_argument("--headless=new")
    for flag in ("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--window-size=1440,1100"):
        options.add_argument(flag)
    driver_path = shutil.which("chromedriver") or "/usr/bin/chromedriver"
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    nav_timeout = max(20, int(job.get("nav_timeout", 120)))
    driver.set_page_load_timeout(nav_timeout)
    driver.set_script_timeout(nav_timeout)
    driver.implicitly_wait(3)
    try:
        for step in steps:
            record = _one_selenium(driver, step, job.get("cookies") or [])
            out["steps"].append(record)
            if record["action"] == "navigate" and record.get("ok"):
                _apply_cookies(driver, job.get("cookies") or [])
        out["final_url"] = driver.current_url
        out["title"] = driver.title
        try:
            out["text"] = (driver.find_element("tag name", "body").text or "")[:20000]
        except Exception:
            out["text"] = ""
        out["cookies"] = _cookie_dicts(driver)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _one_selenium(driver: Any, step: Dict[str, Any], cookies: List[Dict[str, Any]]) -> Dict[str, Any]:
    action = str(step.get("action") or "").strip()
    record: Dict[str, Any] = {"action": action}
    try:
        if action == "navigate":
            url = str(step.get("url") or "")
            if not url.startswith(("http://", "https://", "file://", "data:")):
                raise RuntimeError("only http/https/file/data URLs are allowed")
            driver.get(url)
            record["result"] = {"url": driver.current_url, "title": driver.title}
        elif action == "fill":
            el = _select(driver, str(step.get("selector") or ""), str(step.get("text") or ""))
            el.clear()
            el.send_keys(str(step.get("value") or ""))
            record["result"] = {"filled": True}
        elif action == "click":
            el = _select(driver, str(step.get("selector") or ""), str(step.get("text") or ""), str(step.get("xpath") or ""))
            el.click()
            record["result"] = {"clicked": True, "url": driver.current_url}
        elif action == "extract":
            mode = str(step.get("mode") or "text")
            if mode == "title":
                record["result"] = {"title": driver.title}
            elif mode == "links":
                els = driver.find_elements("tag name", "a")
                record["result"] = [
                    {"text": (e.text or "").strip(), "href": e.get_attribute("href") or ""}
                    for e in els
                ][:int(step.get("limit", 200))]
            else:
                el = _select(driver, str(step.get("selector") or "body")) if step.get("selector") else driver.find_element("tag name", "body")
                if mode == "html":
                    value = el.get_attribute("outerHTML") or ""
                elif mode == "attr":
                    value = el.get_attribute(str(step.get("attr") or "href")) or ""
                else:
                    value = el.text or ""
                if step.get("all"):
                    els = driver.find_elements("css selector", str(step.get("selector") or "body"))
                    value = [e.text if mode == "text" else (e.get_attribute("outerHTML") if mode == "html" else e.get_attribute(str(step.get("attr") or "href")) or "") for e in els]
                record["result"] = value[: int(step.get("limit", 20000))] if isinstance(value, list) else value[:20000]
        elif action == "screenshot":
            path = str(step.get("path") or "")
            if not path:
                raise RuntimeError("screenshot needs a path")
            driver.save_screenshot(path)
            record["result"] = {"path": path}
        elif action == "scroll":
            direction = str(step.get("direction") or "bottom")
            if direction == "top":
                driver.execute_script("window.scrollTo(0,0)")
            elif direction == "bottom":
                driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            else:
                driver.execute_script(f"window.scrollBy(0,{int(step.get('pixels', 600))})")
            record["result"] = {"scrolled": direction}
        elif action == "wait":
            if step.get("selector"):
                from selenium.webdriver.support.ui import WebDriverWait
                WebDriverWait(driver, int(step.get("seconds", 10))).until(
                    lambda d: d.find_elements("css selector", str(step["selector"]))
                )
                record["result"] = {"found": step["selector"]}
            else:
                time.sleep(min(int(step.get("ms", 500)), 15000) / 1000)
                record["result"] = {"waited_ms": int(step.get("ms", 500))}
        elif action == "back":
            driver.back()
            record["result"] = {"url": driver.current_url}
        elif action == "cookies":
            record["result"] = _cookie_dicts(driver)
        elif action == "js":
            record["result"] = driver.execute_script(str(step.get("script") or ""))
        elif action in ("crawl", "rank", "search"):
            query = str(step.get("query") or step.get("q") or "").lower().strip()
            limit = max(1, min(int(step.get("limit", 20)), 100))
            include_text = bool(step.get("include_text", False))
            els = driver.find_elements("tag name", "a")
            seen = set(); scored = []
            for e in els:
                try:
                    href = (e.get_attribute("href") or "").strip()
                    text = (e.text or "").strip()
                except Exception:
                    continue
                if not href.startswith(("http://", "https://")) or href in seen:
                    continue
                seen.add(href)
                score = 1
                hay = (text + " " + href).lower()
                if query:
                    for tok in [t for t in query.split() if t]:
                        if tok in hay:
                            score += 5
                if any(x in href.lower() for x in ["/login", "/signin", "/admin", "/portal", "/api", "/account"]):
                    score += 2
                scored.append({"score": score, "text": text[:200], "href": href})
            scored.sort(key=lambda x: (-x["score"], x["href"]))
            out = {"query": query, "count": len(scored[:limit]), "results": scored[:limit], "title": driver.title, "url": driver.current_url}
            if include_text:
                try:
                    out["page_text"] = (driver.find_element("tag name", "body").text or "")[:8000]
                except Exception:
                    pass
            record["result"] = out
        else:
            raise RuntimeError(f"unknown action: {action}")
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return record


def _run_playwright(job: Dict[str, Any], steps: List[Dict[str, Any]], out: Dict[str, Any]) -> None:
    from playwright.sync_api import sync_playwright
    import shutil

    with sync_playwright() as p:
        launch: Dict[str, Any] = {"headless": job.get("headless", True)}
        chrome = shutil.which("chromium") or "/usr/bin/chromium"
        if __import__("os").path.exists(chrome):
            launch["executable_path"] = chrome
        launch["args"] = ["--no-sandbox", "--disable-dev-shm-usage"]
        browser = p.chromium.launch(**launch)
        context = browser.new_context(
            storage_state=job.get("storage") or None,
            viewport={"width": 1440, "height": 1100},
        )
        page = context.new_page()
        nav_timeout = max(20, int(job.get("nav_timeout", 120)))
        page.set_default_timeout(nav_timeout * 1000)
        page.set_default_navigation_timeout(nav_timeout * 1000)
        try:
            for step in steps:
                out["steps"].append(_one_playwright(page, context, step))
            out["final_url"] = page.url
            out["title"] = page.title()
            out["text"] = (page.locator("body").inner_text() or "")[:20000]
            out["storage"] = context.storage_state()
        finally:
            browser.close()


def _one_playwright(page: Any, context: Any, step: Dict[str, Any]) -> Dict[str, Any]:
    action = str(step.get("action") or "").strip()
    record: Dict[str, Any] = {"action": action}
    try:
        if action == "navigate":
            url = str(step.get("url") or "")
            if not url.startswith(("http://", "https://", "file://", "data:")):
                raise RuntimeError("only http/https/file/data URLs are allowed")
            page.goto(url, wait_until="domcontentloaded")
            record["result"] = {"url": page.url, "title": page.title()}
        elif action == "fill":
            page.locator(str(step.get("selector") or step.get("text") or "")).fill(str(step.get("value") or ""))
            record["result"] = {"filled": True}
        elif action == "click":
            if step.get("text"):
                page.get_by_text(str(step["text"]), exact=False).first.click()
            else:
                page.locator(str(step.get("selector") or step.get("xpath") or "")).first.click()
            record["result"] = {"clicked": True, "url": page.url}
        elif action == "extract":
            mode = str(step.get("mode") or "text")
            if mode == "title":
                record["result"] = {"title": page.title()}
            elif mode == "links":
                record["result"] = [
                    {"text": (a.inner_text() or "").strip(), "href": a.get_attribute("href") or ""}
                    for a in page.locator("a").all()[: int(step.get("limit", 200))]
                ]
            else:
                loc = page.locator(str(step.get("selector") or "body"))
                if mode == "html":
                    value = loc.first.inner_html()
                elif mode == "attr":
                    value = loc.first.get_attribute(str(step.get("attr") or "href")) or ""
                else:
                    value = loc.first.inner_text()
                record["result"] = value[:20000]
        elif action == "screenshot":
            page.screenshot(path=str(step.get("path") or ""))
            record["result"] = {"path": step.get("path")}
        elif action == "scroll":
            page.mouse.wheel(0, int(step.get("pixels", 800)) if step.get("direction") not in ("top", "bottom") else 100000)
            if step.get("direction") == "top":
                page.evaluate("window.scrollTo(0,0)")
            record["result"] = {"scrolled": step.get("direction", "down")}
        elif action == "wait":
            if step.get("selector"):
                page.locator(str(step["selector"])).first.wait_for(state="visible")
                record["result"] = {"found": step["selector"]}
            else:
                page.wait_for_timeout(min(int(step.get("ms", 500)), 15000))
                record["result"] = {"waited_ms": int(step.get("ms", 500))}
        elif action == "back":
            page.go_back()
            record["result"] = {"url": page.url}
        elif action == "cookies":
            try:
                record["result"] = context.cookies()
            except Exception:
                record["result"] = []
        elif action == "js":
            record["result"] = page.evaluate(str(step.get("script") or ""))
        elif action in ("crawl", "rank", "search"):
            record["result"] = _crawl_rank_playwright(page, step)
        else:
            raise RuntimeError(f"unknown action: {action}")
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return record


def _crawl_rank_playwright(page: Any, step: Dict[str, Any]) -> Dict[str, Any]:
    query = str(step.get("query") or step.get("q") or "").lower().strip()
    limit = max(1, min(int(step.get("limit", 20)), 100))
    include_text = bool(step.get("include_text", False))
    items = []
    try:
        links = page.eval_on_selector_all("a[href]", "els => els.map(e => ({text:(e.innerText||'').trim(), href:e.href||'', title:e.title||''}))")
    except Exception:
        links = []
    seen = set()
    scored = []
    for l in links or []:
        href = str(l.get("href") or "").strip()
        text = str(l.get("text") or "").strip()
        if not href.startswith(("http://", "https://")) or href in seen:
            continue
        seen.add(href)
        score = 1
        hay = (text + " " + href + " " + str(l.get("title") or "")).lower()
        if query:
            for tok in [t for t in query.split() if t]:
                if tok in hay:
                    score += 5
        if any(x in href.lower() for x in ["/login", "/signin", "/admin", "/portal", "/api", "/account"]):
            score += 2
        scored.append({"score": score, "text": text[:200], "href": href, "title": str(l.get("title") or "")[:200]})
    scored.sort(key=lambda x: (-x["score"], x["href"]))
    out = {"query": query, "count": len(scored[:limit]), "results": scored[:limit]}
    if include_text:
        try:
            out["page_text"] = (page.locator("body").inner_text() or "")[:8000]
        except Exception:
            pass
    out["title"] = page.title()
    out["url"] = page.url
    return out


def _normalize_action(a: Any) -> Optional[Dict[str, Any]]:
    if isinstance(a, str):
        if a.startswith(("http://", "https://", "file://", "data:")):
            return {"action": "navigate", "url": a}
        return {"action": "extract", "selector": a}
    if not isinstance(a, dict):
        return None
    d = dict(a)
    if d.get("action"):
        return d
    for k in ("navigate", "goto", "open", "url"):
        if k in d and isinstance(d[k], str):
            return {"action": "navigate", "url": d[k], **{x: y for x, y in d.items() if x != k}}
    if "wait" in d:
        w = d["wait"]
        if isinstance(w, dict): return {"action":"wait", **w}
        if isinstance(w, (int,float)): return {"action":"wait","ms":int(w)}
        if isinstance(w, str): return {"action":"wait","selector":w}
    for k in ("click","fill","extract","screenshot","scroll","js","crawl","rank","search"):
        if k in d:
            v=d[k]
            if isinstance(v, dict): return {"action":k, **v}
            if k in ("crawl","rank","search"): return {"action":k, "query": str(v)}
            if k=="click": return {"action":"click","selector":str(v)}
            if k=="fill": return {"action":"fill","value":str(v)}
            if k=="extract": return {"action":"extract","selector":str(v)}
            if k=="screenshot": return {"action":"screenshot","path":str(v)}
            if k=="scroll": return {"action":"scroll","direction":str(v)}
            if k=="js": return {"action":"js","script":str(v)}
    return d


def main() -> None:
    job: Dict[str, Any] = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.loads(sys.stdin.read())
    raw_steps = job.get("actions") or []
    steps: List[Dict[str, Any]] = []
    for a in raw_steps:
        n = _normalize_action(a)
        if n:
            steps.append(n)
    out: Dict[str, Any] = {"ok": False, "engine": "", "steps": [], "warnings": []}
    engine = str(job.get("engine") or "auto").lower()
    order = ["playwright", "selenium"] if engine in ("auto", "playwright") else ["selenium"]
    if engine == "selenium":
        order = ["selenium"]
    for name in order:
        try:
            if name == "playwright":
                _run_playwright(job, steps, out)
            else:
                _run_selenium(job, steps, out)
            out["engine"] = name
            out["ok"] = all(s.get("ok") for s in out["steps"]) if out["steps"] else True
            break
        except Exception as exc:  # noqa: BLE001
            out["steps"] = []
            out["warnings"].append(f"{name} unavailable: {type(exc).__name__}: {str(exc)[:300]}")
    if not out["engine"]:
        out["error"] = "; ".join(out["warnings"]) or "no browser engine available"
    print(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
