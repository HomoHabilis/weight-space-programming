```python
# Error categorization: compile check for all 'error' samples; run stdin ones on public tests to get exception type
errs=[s for s in S if s['outcome']=='error']
res=[]
for s in errs:
    code=last_block(s['text']); pid=s['id']; pr=P[pid]
    try:
        compile(code,'x','exec'); kind=None
    except SyntaxError as e:
        kind='SyntaxError: '+str(e.msg)[:40]
    if kind is None and pr['func'] in (None,'None',''):
        pub=pr['public'] if isinstance(pr['public'],list) else eval(pr['public'])
        with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False) as f: f.write(code); fn=f.name
        kind='ok_on_public'
        for tcase in pub:
            try:
                r=subprocess.run(['python3','-I',fn],input=tcase['input'],capture_output=True,text=True,timeout=6)
                if r.returncode!=0:
                    last=[l for l in r.stderr.strip().split('\n') if l][-1] if r.stderr.strip() else 'rc'
                    kind='RT: '+last[:70]; break
            except subprocess.TimeoutExpired: kind='timeout_public'; break
    elif kind is None: kind='leetcode-runtime?'
    res.append((pid, s['sample'], s['tokens'], kind))
for r in res: print(r)
```