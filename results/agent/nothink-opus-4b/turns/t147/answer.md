```python
# Compare token usage & truncation: control (orig, CP4) vs E4 (x0.75) vs E3 (x1.5) on the same 20 problems
def load_rows_(k): return [json.loads(l) for l in open(f'/work/code_{k}.rows.jsonl')]
R = {name: pd.DataFrame([json.loads(l) for l in open(f'/work/code_{k}.gen.jsonl')]) for name,k in [('E1',8),('E3',10),('E4',11),('orig',12)]}
for name,k in [('E1',8),('E3',10),('E4',11),('orig',12)]:
    rr=pd.DataFrame(load_rows_(k)); R[name]=rr
print(R['orig'].columns.tolist())
for name in ['orig','E4','E3','E1']:
    rr=R[name]; print(name, 'pass', rr.passed.mean(), 'trunc', (~rr.finished.astype(bool)).mean(), 'median tok', rr.tokens.median(), 'mean tok', round(rr.tokens.mean()))
```