```python
# CP7 (V4) paired analysis vs CP6 (V3)
g21=[json.loads(l) for l in open('/work/code_21.gen.jsonl')]; r21=[json.loads(l) for l in open('/work/code_21.rows.jsonl')]
d21=paired(g21,r21)
for nm,dd in [('V3 CP6',d19),('V4 CP7',d21)]:
    print(nm, 'first', dd['first'].mean().round(3), 'final', dd.passed.mean().round(3), '| fixes', ((~dd['first'])&dd.passed).sum(), 'breaks', (dd['first']&~dd.passed).sum(),
          '| trunc', (~dd.fin).sum(), 'no-block', (dd.nb==0).sum(), '| </think>', (dd.think>0).sum(), '| by grp final', dd.groupby('grp').passed.mean().round(2).to_dict(), 'first', dd.groupby('grp')['first'].mean().round(2).to_dict())
print(d21[(d21['first']!=d21.passed)|(~d21.fin)][['id','smp','grp','tok','fin','nb','first','passed','think']])
```