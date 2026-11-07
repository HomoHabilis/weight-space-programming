```python
# Inspect LeetCode 'error' samples: short excerpts of code + test format
lc_err=[s for s in errs if P[s['id']]['func'] not in (None,'None','')]
t0=load_tests(lc_err[0]['id'])[0]; print({k:str(v)[:120] for k,v in t0.items()}, P[lc_err[0]['id']]['func'])
for s in lc_err:
    code=last_block(s['text'])
    print('==',s['id'], s['sample'], s['tokens'], 'func', P[s['id']]['func'], '| ncode lines', code.count('\n'))
    print('   ', repr(code[:200]))
```