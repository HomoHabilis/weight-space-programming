```python
# Summaries: per sample, first block result, any-correct, first-correct position (fraction of thinking chars)
rows=[]
for k,s in enumerate(samples):
    r=tb_res[k]; th,ans=split(s['text'])
    first_ok = next((j for j,g in enumerate(r) if g=='passed'), None)
    rows.append(dict(id=s['id'],diff=s['difficulty'],outcome=s['outcome'],nblk=len(r),
        first=r[0] if r else 'none', last=r[-1] if r else 'none', any_ok=first_ok is not None,
        first_ok_frac=(think_blocks[k][first_ok][0]/len(th)) if first_ok is not None else None))
R=pd.DataFrame(rows)
print(pd.crosstab(R.outcome, R.any_ok))
print(pd.crosstab(R.outcome, R['first']))
print(pd.crosstab(R.outcome, R['last']))
print(R.groupby('outcome').first_ok_frac.median(), R.nblk.describe())
```