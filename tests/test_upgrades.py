import asyncio, json, os, time
from pathlib import Path

from agent.core import ToolRegistry
from agent import auth_vault, blackboard, knowledge_graph as kg, evidence as ev, tool_parsers, oracle


def test_specs_have_new_tools():
    r = ToolRegistry()
    names = {s['function']['name'] for s in r.local_specs()}
    for n in ['http_request','auth_vault','proxy_control','source_index','blackboard','oracle_validate','attack_paths','verify_findings']:
        assert n in names


def test_auth_vault_redacts():
    target = f"test_{int(time.time()*1000)}"
    res = auth_vault.set_profile(target, 'admin', {'authorization':'Bearer secret', 'cookies':{'sid':'abc'}})
    assert res['ok']
    got = auth_vault.get_profile(target, 'admin', include_secrets=False)
    assert 'REDACTED' in json.dumps(got)
    listed = auth_vault.list_profiles(target)
    assert listed['count'] == 1


def test_blackboard_roundtrip():
    target = f"bb_{int(time.time()*1000)}"
    item = blackboard.post(target, 'intent', 'probe /api', source='test')
    items = blackboard.list_items(target, kind='intent')
    assert any(i['id']==item['id'] for i in items)
    blackboard.resolve(target, item['id'], 'done', 'ok')
    summ = blackboard.summary(target)
    assert summ['count'] >= 1


def test_tool_parser_nmap_and_upsert(tmp_path):
    parsed = tool_parsers.parse_nmap('Nmap scan report for 1.2.3.4\n80/tcp open http Apache')
    assert parsed['services'][0]['port'] == 80
    target = f"kg_{int(time.time()*1000)}"
    counts = tool_parsers.upsert_to_graph(target, parsed, source='test')
    assert counts['services'] >= 1
    summ = kg.summary(target)
    assert summ['counts'].get('service',0) >= 1


def test_oracle_reflection():
    res = oracle.run_oracle('reflection_check', url='http://127.0.0.1:9', marker='x')
    assert res['oracle'] == 'reflection_check'


def test_attack_paths_empty():
    target = f"ap_{int(time.time()*1000)}"
    assert kg.rank_attack_paths(target) == []


def test_verify_save_repro(tmp_path):
    target = f"vf_{int(time.time()*1000)}"
    kg.add_finding(target, 'Test finding', 'high', 'details', 'ev1', {'status':'attempted'})
    art = ev.save_repro_artifact(target, 'Test finding', 'print("ok")\n', kind='py')
    assert art['ok'] and Path(art['path']).exists()

