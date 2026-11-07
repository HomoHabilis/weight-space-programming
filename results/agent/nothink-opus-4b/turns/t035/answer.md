```python
# For finished, compiling, wrong samples: do they pass the PUBLIC examples? (verification failure vs hidden-case failure)
import ast as _ast
def pub_tests(pid):
    pub = pm[pid]['public']
    try: L = json.loads(pub)
    except Exception: L = _ast.literal_eval(pub)
    return L
def grade_tests(pid, code, tests):
    func=pm[pid]['func']
    for tst in tests:
        o,m=run_one(code,tst,func)
        if o!='pass': return o
    return 'passed'
w = df[df.finished & (df.compiles==True) & (~df.passed)]
with cf.ThreadPoolExecutor(16) as ex:
    pres = list(ex.map(lambda r: grade_tests(r[0],r[1],pub_tests(r[0])), zip(w.id,w.code)))
w = w.assign(pub=pres)
print(w.groupby(['difficulty','pub']).size())
```