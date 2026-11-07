```python
# Build a local grader to get error messages (stdin and functional), run on 'error' samples
import subprocess, tempfile, concurrent.futures as cf
PRE = "from typing import *\nimport sys, math, collections, itertools, heapq, bisect, functools, string, re\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom functools import *\nfrom math import *\n"
def run_one(code, test, func, timeout=6):
    if test['testtype']=='stdin':
        src = code
        inp = test['input']
    else:
        args = [l for l in test['input'].split('\n')]
        src = PRE + code + "\nimport json as _j\n_a=[_j.loads(l) for l in sys.stdin.read().split('\\n') if l.strip()!='']\nprint(_j.dumps(Solution()."+func+"(*_a)))\n"
        inp = test['input']
    with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,dir='/tmp') as f:
        f.write(src); fn=f.name
    try:
        r = subprocess.run(['python3',fn],input=inp,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return 'timeout',''
    finally:
        os.unlink(fn)
    if r.returncode!=0: return 'error', r.stderr[-300:]
    if test['testtype']=='stdin':
        ok = r.stdout.split()==test['output'].split()
    else:
        try: ok = json.loads(r.stdout.strip().split('\n')[-1])==json.loads(test['output'])
        except Exception as ex: ok = r.stdout.strip()==test['output'].strip()
    return ('pass' if ok else 'wrong'), (r.stdout[:100] if not ok else '')
TESTS = {}
def get_tests(pid):
    if pid not in TESTS: TESTS[pid]=json.load(gzip.open(f'/practice/tests/{pid}.json.gz'))
    return TESTS[pid]
def grade(pid, code, stop_first=True):
    if code is None: return 'no code',''
    func = pm[pid]['func']
    for tst in get_tests(pid):
        o,msg = run_one(code,tst,func)
        if o!='pass': return o,msg
    return 'passed',''
ee = e[e.finished]
with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(lambda r: grade(r[0],r[1]), zip(ee.id,ee.code)))
for (i,r),(o,m) in zip(ee.iterrows(),res):
    print(r.id, r.outcome,'->',o, m.strip().split('\n')[-1][:150] if m else '')
```