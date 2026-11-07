```python
# CP8 paired analysis (fresh: 10 zero + 10 always)
g23=[json.loads(l) for l in open('/work/code_23.gen.jsonl')]; r23=[json.loads(l) for l in open('/work/code_23.rows.jsonl')]
grp={p:'zero' for p in cp8[:10]}; grp.update({p:'always' for p in cp8[10:]})
d23=paired(g23,r23)
print('first', d23['first'].mean(), 'final', d23.passed.mean(), '| fixes', ((~d23['first'])&d23.passed).sum(), 'breaks', (d23['first']&~d23.passed).sum(), '| trunc', (~d23.fin).sum(), 'no-block', (d23.nb==0).sum())
print(d23.groupby('grp')[['first','passed']].mean().round(3))
print(d23[(d23['first']!=d23.passed)|((d23.grp=='always')&~d23.passed)][['id','smp','grp','tok','fin','nb','first','passed','think']])
print('baseline zero/always:', np.mean([s['passed'] for s in S if grp.get(s['id'])=='zero']), np.mean([s['passed'] for s in S if grp.get(s['id'])=='always']))
```