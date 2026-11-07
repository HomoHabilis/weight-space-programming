```python
# Run LC error samples with a simple harness to get exception types
HARN='''
import sys, json
from typing import *
from collections import *
import math, heapq, bisect, itertools, functools, string, re, random, operator
'''
def run_lc(code, func, tests, n=3):
    out=[]
    for tc in tests[:n]:
        args=[json.loads(x) for x in tc['input'].split('\n')]
        prog=HARN+code+f'''
args={json.dumps(args)}
try:
    r=Solution().{func}(*args)
    print(json.dumps(r))
except Exception as e:
    import traceback; print('EXC', type(e).__name__, str(e)[:80])
'''
        with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False) as f: f.write(prog); fn=f.name
        try:
            r=subprocess.run(['python3','-I',fn],capture_output=True,text=True,timeout=6)
            o=(r.stdout.strip().split('\n') or [''])[-1]
            if r.returncode!=0: o='RC '+(r.stderr.strip().split('\n') or [''])[-1][:80]
        except subprocess.TimeoutExpired: o='TIMEOUT'
        out.append((o[:90], tc['output'][:30]))
    return out
for s in lc_err:
    pr=P[s['id']]; pub=pr['public'] if isinstance(pr['public'],list) else eval(pr['public'])
    print(s['id'], s['sample'], run_lc(last_block(s['text']), pr['func'], pub, 2))
```