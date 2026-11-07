```python
# CP6 paired analysis: first block vs final, by group
g19=[json.loads(l) for l in open('/work/code_19.gen.jsonl')]; r19=[json.loads(l) for l in open('/work/code_19.rows.jsonl')]
grp={p:'mixed' for p in cp6[:8]}; grp.update({p:'zero' for p in cp6[8:14]}); grp.update({p:'always' for p in cp6[14:]})
def paired(gen, rows_):
    out=[]
    for g,r in zip(gen,rows_):
        t=g['text']; bs=blocks(t)
        fb=eval_all(g['id'], bs[0]) if bs else False
        out.append(dict(id=g['id'],smp=g['sample'],grp=grp.get(g['id'],'?'),tok=int(r['n_tokens']),fin=r['finished'] in (True,'True'),
                        passed=r['passed'] in (True,'True'),nb=len(bs),first=fb, think=t.count('</think>')))
    return pd.DataFrame(out)
d19=paired(g19,r19)
print(d19.groupby('grp')[['first','passed']].mean().round(2), '\n', d19.groupby('grp').size())
print('fixes', ((~d19['first'])&d19.passed).sum(), 'breaks', (d19['first']&~d19.passed).sum(), '| first-block pass', d19['first'].mean(), 'final', d19.passed.mean())
print('baseline by group:', {gname: round(np.mean([s['passed'] for s in S if grp.get(s['id'])==gname]),2) for gname in ['mixed','zero','always']})
print('nblocks dist', Counter(d19.nb), '| answers with </think>', (d19.think>0).sum(), '| trunc', (~d19.fin).sum())
print(d19[(d19['first']!=d19.passed)][['id','smp','grp','tok','fin','nb','first','passed','think']])
```