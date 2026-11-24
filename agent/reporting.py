"""Generate pentest reports from graph + evidence ledger (md/html/json/csv)."""

from __future__ import annotations

import csv
import html
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from agent import evidence, knowledge_graph as kg


SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _ts(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(ts)))
    except Exception:
        return "-"


def _sorted_findings(target: str) -> List[Dict[str, Any]]:
    findings = kg.query(target, kind="finding", limit=500)
    findings.sort(key=lambda n: (
        SEV_ORDER.get(str(n.get("attrs", {}).get("severity", "info")).lower(), 9),
        0 if str(n.get("attrs", {}).get("status", "attempted")).lower() == "proved" else 1,
    ))
    return findings


def build_report_data(target: str) -> Dict[str, Any]:
    g = kg.summary(target)
    ev = evidence.summary(target)
    findings = _sorted_findings(target)
    ev_entries = evidence.read_entries(target, limit=500)
    counts = g.get("counts", {})
    proved = sum(1 for n in findings if str(n.get("attrs", {}).get("status", "attempted")).lower() == "proved")
    attempted = sum(1 for n in findings if str(n.get("attrs", {}).get("status", "attempted")).lower() != "proved")
    finding_rows = []
    for i, n in enumerate(findings, 1):
        a = n.get("attrs", {})
        finding_rows.append({
            "index": i,
            "id": n.get("key") or n.get("id"),
            "title": a.get("title") or n.get("id") or "Finding",
            "severity": str(a.get("severity", "info")).lower(),
            "status": str(a.get("status", "attempted")).lower(),
            "asset": a.get("asset") or "",
            "details": a.get("details") or "",
            "repro": a.get("repro") or a.get("reproduction") or "",
            "impact": a.get("impact") or "",
            "remediation": a.get("remediation") or "",
            "evidence_ref": a.get("evidence_ref") or "",
            "confidence": a.get("confidence"),
            "sources": n.get("sources") or [],
        })
    trace = []
    for e in ev_entries:
        trace.append({
            "id": e.get("id"),
            "ts": e.get("ts"),
            "time_utc": _ts(e.get("ts")),
            "tool": e.get("tool"),
            "ok": bool(e.get("ok")),
            "source": e.get("source"),
            "tags": e.get("tags") or [],
            "result_sha256": e.get("result_sha256"),
            "result_bytes": e.get("result_bytes"),
            "result_preview": str(e.get("result_preview") or "")[:500],
        })
    attack_paths = []
    blackboard_summary = {}
    try:
        attack_paths = kg.rank_attack_paths(target, max_paths=10, max_depth=5)
    except Exception:
        pass
    try:
        from agent import blackboard as bb
        blackboard_summary = bb.summary(target)
    except Exception:
        pass
    return {
        "target": g.get("target", target),
        "slug": g.get("slug", ""),
        "generated_at": _ts(time.time()),
        "generated_ts": time.time(),
        "summary": {
            "hosts": counts.get("host", 0),
            "services": counts.get("service", 0),
            "endpoints": counts.get("endpoint", 0),
            "findings": counts.get("finding", 0),
            "proved": proved,
            "attempted_or_unverified": attempted,
            "evidence_entries": ev.get("count", 0),
            "evidence_ok": ev.get("ok", 0),
            "evidence_failed": ev.get("failed", 0),
        },
        "graph_path": g.get("path", ""),
        "evidence_path": ev.get("path", ""),
        "findings": finding_rows,
        "attack_surface": {
            "hosts": [n.get("attrs", {}) for n in kg.query(target, kind="host", limit=500)],
            "services": [n.get("attrs", {}) for n in kg.query(target, kind="service", limit=1000)],
            "endpoints": [n.get("attrs", {}) for n in kg.query(target, kind="endpoint", limit=1000)],
        },
        "attack_paths": attack_paths,
        "blackboard": blackboard_summary,
        "execution_trace": trace,
        "evidence_by_tool": ev.get("by_tool") or {},
    }


def build_markdown(target: str) -> str:
    d = build_report_data(target)
    s = d["summary"]
    lines: List[str] = []
    lines.append(f"# Pentest Report - {d['target']}")
    lines.append("")
    lines.append(f"- Generated: {d['generated_at']}")
    lines.append(f"- Target slug: `{d['slug']}`")
    lines.append(f"- Knowledge graph: `{d['graph_path']}`")
    lines.append(f"- Evidence ledger: `{d['evidence_path']}`")
    lines.append(f"- Evidence entries: {s['evidence_entries']} (ok={s['evidence_ok']}, failed={s['evidence_failed']})")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Discovered **{s['hosts']} hosts**, **{s['services']} services**, **{s['endpoints']} endpoints**, **{s['findings']} findings**.")
    lines.append(f"Findings status: **{s['proved']} proved**, **{s['attempted_or_unverified']} attempted/unverified**. A finding is only proved with reproducible evidence.")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not d["findings"]:
        lines.append("_No findings recorded yet._")
    for f in d["findings"]:
        lines.append(f"### {f['index']}. [{f['severity'].upper()}] [{f['status'].upper()}] {f['title']}")
        if f.get("asset"):
            lines.append(f"- Asset: `{f['asset']}`")
        if f.get("confidence") is not None:
            lines.append(f"- Confidence: `{f['confidence']}`")
        if f.get("details"):
            lines.append(f"\n{f['details']}")
        if f.get("repro"):
            lines.append(f"\n**Reproduction**\n\n{f['repro']}")
        if f.get("impact"):
            lines.append(f"\n**Impact**\n\n{f['impact']}")
        if f.get("remediation"):
            lines.append(f"\n**Remediation**\n\n{f['remediation']}")
        if f.get("evidence_ref"):
            lines.append(f"\n- Evidence ref: `{f['evidence_ref']}`")
        lines.append("")
    lines.append("## Attack Surface")
    lines.append("")
    for section, key, fmt in [("Hosts", "hosts", lambda a: f"- `{a.get('host') or a.get('id','')}`" + (f" - {a.get('notes')}" if a.get('notes') else "")),
                              ("Services", "services", lambda a: f"- `{a.get('host')}:{a.get('port')}/{a.get('proto','tcp')}`" + (f" ({a.get('name') or a.get('product')})" if (a.get('name') or a.get('product')) else "")),
                              ("Endpoints", "endpoints", lambda a: f"- {a.get('url') or a.get('id','')}")]:
        lines.append(f"### {section}")
        lines.append("")
        items = d["attack_surface"].get(key) or []
        for a in items[:300]:
            lines.append(fmt(a))
        if not items:
            lines.append("_None_")
        lines.append("")
    lines.append("## Attack Paths")
    lines.append("")
    aps = d.get("attack_paths") or []
    if not aps:
        lines.append("_No ranked attack paths yet._")
    for pth in aps[:10]:
        seq = " -> ".join(f"{n.get('kind')}:{n.get('label')}" for n in pth.get("nodes", []))
        lines.append(f"- score={pth.get('score')}: {seq}")
    lines.append("")
    lines.append("## Execution Trace (recent evidence)")
    lines.append("")
    lines.append("| Time (UTC) | ID | Tool | OK | SHA-256 | Preview |")
    lines.append("|---|---|---|---:|---|---|")
    for e in d["execution_trace"][-100:]:
        preview = str(e.get("result_preview") or "").replace("\n", " ")[:120].replace("|", "\\|")
        lines.append(f"| {e.get('time_utc','')} | `{str(e.get('id',''))[:12]}` | `{e.get('tool','')}` | {'yes' if e.get('ok') else 'no'} | `{str(e.get('result_sha256',''))[:12]}` | {preview} |")
    if not d["execution_trace"]:
        lines.append("| - | - | - | - | - | _No evidence recorded_ |")
    lines.append("")
    return "\n".join(lines)


def build_html(target: str) -> str:
    d = build_report_data(target)
    s = d["summary"]
    def esc(x: Any) -> str: return html.escape(str(x or ""))

    finding_blocks = []
    for f in d["findings"]:
        body = f"<pre>{esc(f.get('details',''))}</pre>"
        if f.get("repro"):
            body += f"<h4>Reproduction</h4><pre>{esc(f['repro'])}</pre>"
        if f.get("impact"):
            body += f"<h4>Impact</h4><pre>{esc(f['impact'])}</pre>"
        if f.get("remediation"):
            body += f"<h4>Remediation</h4><pre>{esc(f['remediation'])}</pre>"
        meta = []
        if f.get("asset"): meta.append(f"Asset: <code>{esc(f['asset'])}</code>")
        if f.get("evidence_ref"): meta.append(f"Evidence: <code>{esc(f['evidence_ref'])}</code>")
        if f.get("confidence") is not None: meta.append(f"Confidence: <code>{esc(f['confidence'])}</code>")
        if meta: body = "<p>" + " | ".join(meta) + "</p>" + body
        finding_blocks.append(f"<section class='finding'><h3>{f['index']}. [{esc(f['severity'].upper())}] [{esc(f['status'].upper())}] {esc(f['title'])}</h3>{body}</section>")

    rows = []
    for e in d["execution_trace"][-150:]:
        rows.append("<tr><td>{}</td><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
            esc(e.get("time_utc")), esc(str(e.get("id",""))[:12]), esc(e.get("tool")), "yes" if e.get("ok") else "no",
            esc(str(e.get("result_sha256",""))[:12]), esc(str(e.get("result_preview","")).replace("\n"," ")[:160])
        ))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Report {esc(d['target'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px;font-size:13px;vertical-align:top}} pre{{white-space:pre-wrap;background:#111;color:#eee;padding:12px;border-radius:8px}} .finding{{margin:16px 0;padding:8px 0;border-bottom:1px solid #eee}} code{{background:#f3f3f3;padding:1px 4px;border-radius:4px}}</style>
</head><body>
<h1>Pentest Report - {esc(d['target'])}</h1>
<p>Generated {esc(d['generated_at'])} | Graph <code>{esc(d['graph_path'])}</code></p>
<h2>Summary</h2><ul>
<li>Hosts: {s['hosts']}</li><li>Services: {s['services']}</li><li>Endpoints: {s['endpoints']}</li><li>Findings: {s['findings']}</li>
<li>Proved: {s['proved']} | Attempted/unverified: {s['attempted_or_unverified']}</li>
<li>Evidence: {s['evidence_entries']} (ok={s['evidence_ok']}, failed={s['evidence_failed']})</li>
</ul>
<h2>Findings</h2>{''.join(finding_blocks) or '<p><em>No findings recorded yet.</em></p>'}
<h2>Execution Trace</h2>
<table><thead><tr><th>Time</th><th>ID</th><th>Tool</th><th>OK</th><th>SHA-256</th><th>Preview</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan=6><em>None</em></td></tr>'}</tbody></table>
</body></html>"""


def build_json(target: str) -> str:
    return json.dumps(build_report_data(target), indent=2, ensure_ascii=False)


def build_csv(target: str) -> str:
    d = build_report_data(target)
    out = io.StringIO()
    fieldnames = ["index", "id", "title", "severity", "status", "asset", "confidence", "evidence_ref", "details", "repro", "impact", "remediation", "sources"]
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for f in d["findings"]:
        row = dict(f)
        row["sources"] = ";".join(row.get("sources") or [])
        w.writerow({k: row.get(k, "") for k in fieldnames})
    return out.getvalue()


def write_report(target: str, fmt: str = "markdown", out: str = "") -> Dict[str, Any]:
    fmt = (fmt or "markdown").lower().strip()
    if fmt in ("md", "markdown"):
        content = build_markdown(target); ext = ".md"
    elif fmt in ("html", "htm"):
        content = build_html(target); ext = ".html"
    elif fmt == "json":
        content = build_json(target); ext = ".json"
    elif fmt == "csv":
        content = build_csv(target); ext = ".csv"
    else:
        raise ValueError("fmt must be markdown, html, json, or csv")
    if out:
        path = Path(out).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    else:
        path = kg.target_dir(target) / f"report{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "format": fmt, "bytes": len(content.encode("utf-8")), "target": target}
