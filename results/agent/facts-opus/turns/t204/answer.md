```python
# Check cross-talk: cosine between logit directions of the 30 first-token targets
t0=np.array(sorted(set(R.tgt.values[P0]))); D0=np.stack([udir(t)[0] for t in t0])
Cc=D0@D0.T; off=Cc[~np.eye(len(t0),dtype=bool)]
print(len(t0),'cos offdiag mean',off.mean().round(3),'max',off.max().round(3))
# random tokens
rt=np.random.RandomState(0).choice(248000,500); Dr=np.stack([udir(t)[0] for t in rt]); print('cos target vs random mean',(D0@Dr.T).mean().round(3),'p99',np.percentile(D0@Dr.T,99).round(3))
```