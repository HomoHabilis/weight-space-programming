```python
# Build a local grader (driver script run in a subprocess); validate on sample final answers
driver = r'''
import sys, json, gzip, io, signal, math
code_path, pid, mode, func = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
code = open(code_path).read()
tests = json.load(gzip.open(f"/practice/tests/{pid}.json.gz"))
PRE = "from typing import *\nimport sys, math, collections, itertools, heapq, bisect, functools, string, re, random\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom functools import *\nfrom math import *\nsys.setrecursionlimit(1000000)\n"
class TO(Exception): pass
def h(s,f): raise TO()
signal.signal(signal.SIGALRM, h)
def eq_tok(a,b):
    a=a.split(); b=b.split()
    if len(a)!=len(b): return False
    for x,y in zip(a,b):
        if x==y: continue
        try:
            if abs(float(x)-float(y))<=1e-6*max(1,abs(float(y))): continue
        except: pass
        return False
    return True
res="passed"
for t in tests:
    try:
        signal.alarm(6)
        if mode=="stdin":
            inp = io.TextIOWrapper(io.BytesIO(t["input"].encode()))
            out = io.StringIO()
            sys.stdin = inp; old=sys.stdout; sys.stdout = out
            g={"__name__":"__main__"}
            try:
                exec(PRE+code, g)
            except SystemExit: pass
            finally:
                sys.stdout = old
            signal.alarm(0)
            if not eq_tok(out.getvalue(), t["output"]): res="wrong"; break
        else:
            g={"__name__":"__main__"}
            exec(PRE+code, g)
            args=[json.loads(l) for l in t["input"].split("\n") if l.strip()!=""]
            r = getattr(g["Solution"](), func)(*args)
            signal.alarm(0)
            exp=json.loads(t["output"])
            if isinstance(r, tuple): r=list(r)
            if r!=exp:
                if isinstance(exp,float) and isinstance(r,(int,float)) and abs(r-exp)<1e-6: continue
                res="wrong"; break
    except TO:
        res="timeout"; break
    except BaseException as e:
        signal.alarm(0)
        sys.stdout=sys.__stdout__
        res="error"; break
sys.stdout=sys.__stdout__
print("RESULT:"+res)
'''
open('/work/driver.py','w').write(driver)
import subprocess, tempfile, hashlib
P={p['id']:p for p in probs}
def last_code(ans):
    blocks=re.findall(BT+r'python\n(.*?)'+BT, ans, re.S)
    return blocks[-1] if blocks else None
def grade(code, pid, total_timeout=120):
    p=P[pid]; mode='stdin' if p['platform']=='atcoder' or not p['starter'] else 'func'
    fn=f"/tmp/c_{hashlib.md5((code+pid).encode()).hexdigest()}.py"
    open(fn,'w').write(code)
    try:
        r=subprocess.run(['python','/work/driver.py',fn,pid,mode,p['func'] or 'x'],capture_output=True,text=True,timeout=total_timeout)
        m=re.search(r'RESULT:(\w+)', r.stdout)
        return m.group(1) if m else 'error'
    except subprocess.TimeoutExpired:
        return 'timeout'
import time
t0=time.time(); agree=0; tot=0; mism=[]
for s in samples[:60]:
    th,ans=split(s['text']); c=last_code(ans)
    if c is None: continue
    g=grade(c,s['id']); tot+=1; agree+= (g=='passed')==s['passed']
    if (g=='passed')!=s['passed']: mism.append((s['id'],s['sample'],g,s['outcome']))
print(agree,tot,time.time()-t0, mism)
```