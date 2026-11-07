```python
# How did truncated answers end? (mid-block or in prose) and what does 'no code' case look like
for i,(g,r) in enumerate(zip(g17,r17)):
    if r['finished'] in (True,'True'): continue
    t=g['text']; nopen=t.count(BT+'python'); nf=t.count(BT)
    mid = (nf%2==1)
    print(g['id'], g['sample'], 'passed', r['passed'], '| ends mid-block', mid, '| code field len', len(r.get('code') or ''), '| tail:', esc(repr(t[-90:])))
```