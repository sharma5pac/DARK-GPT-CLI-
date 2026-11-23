import json, os, shutil, tempfile
from pathlib import Path

# run in temp HOME to avoid polluting real config
TMP = Path(tempfile.mkdtemp(prefix="dgtest_"))
os.environ["HOME"] = str(TMP)
(TMP/".config/darkgpt").mkdir(parents=True, exist_ok=True)

from agent import evidence, knowledge_graph as kg, reporting, planner
from agent.subagents import parse_role_output

def ok(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond: raise SystemExit(1)

# Evidence id + ref lookup
e = evidence.record("t1", tool="shell", args={"cmd":"echo hi"}, result="hello", ok=True)
ok("evidence has id", bool(e.get("id")) and len(e["id"]) >= 8)
ok("evidence has sha256", len(e.get("result_sha256","")) == 64)
ok("evidence has_ref by id", evidence.has_ref("t1", e["id"]))
ok("evidence has_ref by sha prefix", evidence.has_ref("t1", e["result_sha256"][:12]))

# Proof gate: proved without evidence_ref must be downgraded
f = kg.add_finding("t1", "SQLi", "high", "details", evidence_ref="", attrs={"status":"proved"}, source="test")
ok("proof gate downgrades", f["attrs"]["status"] == "attempted")
ok("proof gate reason", bool(f["attrs"].get("proof_gate_blocked")))

# Proved with valid evidence ref allowed
f2 = kg.add_finding("t1", "SQLi2", "high", "details", evidence_ref=e["id"], attrs={"status":"proved"}, source="test")
ok("proved with evidence_ref", f2["attrs"]["status"] == "proved")

# Report formats
md = reporting.build_markdown("t1")
js = reporting.build_json("t1")
csv = reporting.build_csv("t1")
ok("report md has proved/attempted", "PROVED" in md and "ATTEMPTED" in md)
obj = json.loads(js)
ok("report json has execution_trace", isinstance(obj.get("execution_trace"), list))
ok("report json has findings", len(obj.get("findings")) >= 1)
ok("report csv header", csv.splitlines()[0].startswith("index,id,title"))
rp = reporting.write_report("t1", fmt="json")
ok("report write json", Path(rp["path"]).exists())
rp2 = reporting.write_report("t1", fmt="csv")
ok("report write csv", Path(rp2["path"]).exists())

# Planner
planner.set_objective("t1", "test objective")
planner.add_hypothesis("t1", "maybe idor", status="open", confidence=0.6)
planner.add_task("t1", "check id", status="todo")
planner.record_observation("t1", tool="shell", observed="403", hypothesis_delta="waf", next_action="mutate")
planner.record_dead_end("t1", "same 403 on /api", reason="waf", pivot="try other endpoint")
ps = planner.summary("t1")
ok("planner objective", ps["objective"] == "test objective")
ok("planner hypothesis", len(ps["open_hypotheses"]) >= 1)
ok("planner observation", len(ps["recent_observations"]) >= 1)
ok("planner dead_end", len(ps["dead_ends"]) >= 1)

# Subagent schema parsing
parsed, schema_ok = parse_role_output("recon", '```json\n{"hosts":[{"host":"a.com"}],"services":[],"endpoints":[],"findings":[],"next_step":"x"}\n```')
ok("recon schema parse", schema_ok and parsed["hosts"][0]["host"] == "a.com")
parsed2, ok2 = parse_role_output("recon", "no json here")
ok("recon schema fail clean", (not ok2) and parsed2 is None)

# KG export
exp = kg.export_graph("t1", "json")
ok("kg export json", exp["format"] == "json" and "graph" in exp)
dot = kg.export_graph("t1", "dot")
ok("kg export dot", dot["format"] == "dot" and Path(dot["path"]).exists())

# KG v2 nodes
kg.add_credential("t1", "api", kind="token", attrs={"secret":"supersecret"}, source="test")
kg.add_auth_state("t1", "a.com", "admin", source="test")
kg.add_param("t1", "https://a.com/api", "id", "query", source="test")
kg.add_fingerprint("t1", "https://a.com", "403+cf", source="test")
s = kg.summary("t1")
ok("kg v2 kinds", s["counts"].get("credential",0)>=1 and s["counts"].get("auth_state",0)>=1 and s["counts"].get("param",0)>=1 and s["counts"].get("fingerprint",0)>=1)

shutil.rmtree(TMP, ignore_errors=True)
print("ALL PILLAR TESTS PASSED")
