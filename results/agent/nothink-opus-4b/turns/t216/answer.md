```python
# Quick diagnosis of mixed-problem failures: public-test behavior (AtCoder ones)
mixed_p=[p for p in base if 0<sum(base[p])<3 and P[p]['platform']=='atcoder']
out=[]
for p in mixed_p:
    pub=P[p]['public'] if isinstance(P[p]['public'],list) else eval(P[p]['public'])
    for s in [x for x in S if x['id']==p and x['outcome']=='wrong']:
        r=run_stdin(last_block(s['text']), pub)
        out.append((p, s['sample'], s['tokens'], r))
print(Counter(o[3].split('@')[0] for o in out))
for o in out[:30]: print(o)
```