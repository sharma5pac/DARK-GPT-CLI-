import asyncio
from agent.core import DEFAULT_MAX_STEPS, DEFAULT_TOOLS_PER_TURN
from agent.jobs import JobManager
from agent.browser import run as browser_run


def test_jobs():
    async def _run():
        jm = JobManager()
        a = await jm.start_shell('echo hello-job')
        b = await jm.start_script('python', 'print("pyjob")')
        w = await jm.wait([a['job']['id'], b['job']['id']], timeout=20)
        assert w['ok'] and w['done']
        by = {j['id']: j for j in w['jobs']}
        assert 'hello-job' in by[a['job']['id']]['output']
        assert 'pyjob' in by[b['job']['id']]['output']
        assert jm.log(a['job']['id'])['ok']
        print('PASS jobs')
    asyncio.run(_run())


def test_browser_norm_crawl():
    r = browser_run([
        {"navigate": "data:text/html,<body><a href='https://ex.com/admin'>Admin</a><a href='https://ex.com/x'>X</a></body>"},
        {"crawl": {"query": "admin", "limit": 5}},
    ], engine='selenium', timeout=60)
    assert r.get('ok'), r
    assert r['steps'][1]['result']['results'][0]['href'] == 'https://ex.com/admin'
    r2 = browser_run([{"navigate": "data:text/html,<title>T</title><body>Hi</body>"}, {"wait": {"ms": 10}}], engine='selenium', timeout=60)
    assert r2.get('ok')
    print('PASS browser')


def test_budget_defaults():
    assert DEFAULT_TOOLS_PER_TURN >= 40
    assert DEFAULT_MAX_STEPS >= 28
    print('PASS budget defaults')


def main():
    test_jobs()
    test_browser_norm_crawl()
    test_budget_defaults()
    print('ALL NEW FEATURE TESTS PASSED')


if __name__ == '__main__':
    main()
