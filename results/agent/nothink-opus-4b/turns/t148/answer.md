```python
for name in ['orig','E4','E3','E1']:
    rr=R[name]; print(name, 'pass', rr.passed.mean(), 'trunc', round((~rr.finished.astype(bool)).mean(),3), 'median tok', rr.n_tokens.median(), 'mean tok', round(rr.n_tokens.mean()))
# 'why' field for failures in orig
print(R['orig'].why.value_counts().head(10))
```