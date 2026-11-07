```python
# LeetCode failure analysis: wrong samples - do they fail public examples? any stdin-reading / test code?
lc=[s for s in S if P[s['id']]['func'] not in (None,'None','')]
print('LC samples', len(lc), Counter(s['outcome'] for s in lc))
cnt=Counter()
for s in lc:
    if s['outcome']!='wrong': continue
    code=last_block(s['text']); pr=P[s['id']]; pub=pr['public'] if isinstance(pr['public'],list) else eval(pr['public'])
    rr=run_lc(code, pr['func'], pub, 3)
    ok=all(o.strip()==json.dumps(json.loads(e)) if not o.startswith(('EXC','RC','TIMEOUT')) else False for o,e in rr) if True else None
    cnt['pub_pass' if ok else 'pub_fail']+=1
    cnt['has_main' if '__main__' in code or 'input(' in code else 'no_main']+=1
print(cnt)
```