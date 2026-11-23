"""Generate markdown/HTML pentest reports from graph + evidence ledger."""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from agent import evidence, knowledge_graph as kg


def _ts(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(ts)))
    except Exception:
        return "-"


def build_markdown(target: str) -> str:
    g = kg.summary(target)
    ev = evidence.summary(target)
    findings = kg.query(target, kind="finding", limit=200)
    hosts = kg.query(target, kind="host", limit=200)
    services = kg.query(target, kind="service", limit=500)
    endpoints = kg.query(target, kind="endpoint", limit=500)
    ev_entries = evidence.read_entries(target, limit=200)

    lines: List[str] = []
    lines.append(f"# Pentest Report — {g.get('target', target)}")
    lines.append("")
    lines.append(f"- Generated: {_ts(time.time())}")
    lines.append(f"- Target slug: `{g.get('slug','')}`")
    lines.append(f"- Knowledge graph: `{g.get('path','')}`")
    lines.append(f"- Evidence ledger: `{ev.get('path','')}`")
    lines.append(f"- Evidence entries: {ev.get('count',0)} (ok={ev.get('ok',0)}, failed={ev.get('failed',0)})")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    counts = g.get("counts", {})
    lines.append(f"Discovered **{counts.get('host',0)} hosts**, **{counts.get('service',0)} services**, **{counts.get('endpoint',0)} endpoints**, **{counts.get('finding',0)} findings**.")
    lines.append("")
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings_sorted = sorted(findings, key=lambda n: sev_order.get(str(n.get("attrs", {}).get("severity", "info")).lower(), 9))
    proved = sum(1 for n in findings_sorted if str(n.get("attrs",{}).get("status","attempted")).lower() == "proved")
    attempted = sum(1 for n in findings_sorted if str(n.get("attrs",{}).get("status","attempted")).lower() != "proved")
    lines.append(f"Findings status: **{proved} proved**, **{attempted} attempted/unverified**. A finding is only proved with reproducible evidence.")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not findings_sorted:
        lines.append("_No findings recorded yet._")
    else:
        for i, n in enumerate(findings_sorted, 1):
            a = n.get("attrs", {})
            sev = str(a.get("severity", "info")).upper()
            status = str(a.get("status", "attempted")).upper()
            title = a.get("title") or n.get("id") or "Finding"
            lines.append(f"### {i}. [{sev}] [{status}] {title}")
            if a.get("asset"):
                lines.append(f"- Asset: `{a['asset']}`")
            if a.get("details"):
                lines.append("")
                lines.append(str(a["details"]))
            if a.get("repro") or a.get("reproduction"):
                lines.append("\n**Reproduction**\n")
                lines.append(str(a.get("repro") or a.get("reproduction")))
            if a.get("impact"):
                lines.append(f"\n**Impact**\n\n{a['impact']}")
            if a.get("remediation"):
                lines.append(f"\n**Remediation**\n\n{a['remediation']}")
            if a.get("evidence_ref"):
                lines.append(f"\n- Evidence ref: `{a['evidence_ref']}`")
            lines.append("")
    lines.append("## Attack Surface")
    lines.append("")
    lines.append("### Hosts")
    lines.append("")
    for n in hosts[:200]:
        a = n.get("attrs", {})
        lines.append(f"- `{a.get('host') or n.get('id')}`" + (f" — {a.get('notes')}" if a.get("notes") else ""))
    if not hosts:
        lines.append("_None_")
    lines.append("")
    lines.append("### Services")
    lines.append("")
    for n in services[:300]:
        a = n.get("attrs", {})
        name = a.get("name") or a.get("product") or ""
        extra = f" ({name})" if name else ""
        lines.append(f"- `{a.get('host')}:{a.get('port')}/{a.get('proto','tcp')}`{extra}")
    if not services:
        lines.append("_None_")
    lines.append("")
    lines.append("### Endpoints")
    lines.append("")
    for n in endpoints[:300]:
        a = n.get("attrs", {})
        lines.append(f"- {a.get('url') or n.get('id')}")
    if not endpoints:
        lines.append("_None_")
    lines.append("")
    lines.append("## Evidence (recent)")
    lines.append("")
    lines.append("| Time (UTC) | Tool | OK | SHA-256 | Preview |")
    lines.append("|---|---|---:|---|---|")
    for e in ev_entries[-100:]:
        preview = str(e.get("result_preview") or "").replace("\n", " ")[:120].replace("|", "\\|")
        lines.append(f"| {_ts(e.get('ts'))} | `{e.get('tool','')}` | {'yes' if e.get('ok') else 'no'} | `{str(e.get('result_sha256',''))[:12]}` | {preview} |")
    if not ev_entries:
        lines.append("| - | - | - | - | _No evidence recorded_ |")
    lines.append("")
    lines.append("## Evidence Tool Counts")
    lines.append("")
    for tool, count in (ev.get("by_tool") or {}).items():
        lines.append(f"- `{tool}`: {count}")
    if not ev.get("by_tool"):
        lines.append("_None_")
    lines.append("")
    return "\n".join(lines)


def build_html(target: str) -> str:
    md = build_markdown(target)
    # Minimal, dependency-free HTML rendering: escaped markdown in <pre> + simple heading extraction isn't great;
    # render a clean HTML report directly.
    g = kg.summary(target)
    ev = evidence.summary(target)
    findings = kg.query(target, kind="finding", limit=200)
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings_sorted = sorted(findings, key=lambda n: sev_order.get(str(n.get("attrs", {}).get("severity", "info")).lower(), 9))
    def esc(x: Any) -> str: return html.escape(str(x or ""))

    rows = []
    for e in evidence.read_entries(target, limit=100):
        rows.append("<tr><td>{}</td><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
            esc(_ts(e.get("ts"))), esc(e.get("tool")), "yes" if e.get("ok") else "no",
            esc(str(e.get("result_sha256", ""))[:12]),
            esc(str(e.get("result_preview", "")).replace("\n", " ")[:160]),
        ))
    finding_blocks = []
    for i, n in enumerate(findings_sorted, 1):
        a = n.get("attrs", {})
        status = esc(str(a.get("status", "attempted")).upper())
        sev = esc(str(a.get("severity","info")).upper())
        body = f"<pre>{esc(a.get('details',''))}</pre>"
        if a.get("repro") or a.get("reproduction"):
            body += f"<h4>Reproduction</h4><pre>{esc(a.get('repro') or a.get('reproduction'))}</pre>"
        if a.get("impact"):
            body += f"<h4>Impact</h4><pre>{esc(a.get('impact'))}</pre>"
        if a.get("remediation"):
            body += f"<h4>Remediation</h4><pre>{esc(a.get('remediation'))}</pre>"
        if a.get("evidence_ref"):
            body += f"<p>Evidence ref: <code>{esc(a.get('evidence_ref'))}</code></p>"
        finding_blocks.append(f"<section class='finding'><h3>{i}. [{sev}] [{status}] {esc(a.get('title') or n.get('id'))}</h3>{body}</section>")
    counts = g.get("counts", {})
    proved = sum(1 for n in findings_sorted if str(n.get("attrs",{}).get("status","attempted")).lower() == "proved")
    attempted = sum(1 for n in findings_sorted if str(n.get("attrs",{}).get("status","attempted")).lower() != "proved")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Report {esc(target)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px;font-size:13px;vertical-align:top}} pre{{white-space:pre-wrap;background:#111;color:#eee;padding:12px;border-radius:8px}} .finding{{margin:16px 0}} code{{background:#f3f3f3;padding:1px 4px;border-radius:4px}}</style>
</head><body>
<h1>Pentest Report — {esc(g.get('target',target))}</h1>
<p>Generated {_ts(time.time())} · Evidence {ev.get('count',0)} entries · Graph {esc(g.get('path',''))}</p>
<h2>Summary</h2><ul>
<li>Hosts: {counts.get('host',0)}</li><li>Services: {counts.get('service',0)}</li><li>Endpoints: {counts.get('endpoint',0)}</li><li>Findings: {counts.get('finding',0)}</li>
<li>Proved: {proved} · Attempted/unverified: {attempted}</li>
</ul>
<h2>Findings</h2>{''.join(finding_blocks) or '<p><em>No findings recorded yet.</em></p>'}
<h2>Recent Evidence</h2>
<table><thead><tr><th>Time</th><th>Tool</th><th>OK</th><th>SHA-256</th><th>Preview</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan=5><em>None</em></td></tr>'}</tbody></table>
</body></html>"""


def write_report(target: str, fmt: str = "markdown", out: str = "") -> Dict[str, Any]:
    fmt = (fmt or "markdown").lower().strip()
    if fmt in ("md", "markdown"):
        content = build_markdown(target)
        ext = ".md"
    elif fmt in ("html", "htm"):
        content = build_html(target)
        ext = ".html"
    else:
        raise ValueError("fmt must be markdown or html")
    if out:
        path = Path(out).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    else:
        path = kg.target_dir(target) / f"report{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "format": fmt, "bytes": len(content.encode("utf-8")), "target": target}