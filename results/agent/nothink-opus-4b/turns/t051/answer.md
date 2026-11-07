```python
# Inspect a meandering truncated sample (3647) at several points
r = df[(df.id=='3647')].iloc[0]; t=r.text
print(pm['3647']['title'], r.outcome, len(t), r.n_wait, r.ngrep)
for frac in [0.05, 0.3, 0.6, 0.9]:
    k=int(len(t)*frac); print('----',frac); print(t[k:k+700])
```