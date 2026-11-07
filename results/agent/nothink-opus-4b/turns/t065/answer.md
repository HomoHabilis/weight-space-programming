```python
# Loop samples: problems and pass rate of their sibling samples
lp_ids = df[df.ngrep>0.5]
for i,r in lp_ids.iterrows():
    sib = df[(df.id==r.id)&(df.index!=i)]
    print(r.id, r.difficulty, r.outcome, r.finished, 'siblings pass', sib.passed.sum(),'/',len(sib), 'sib outcomes', list(sib.outcome))
```