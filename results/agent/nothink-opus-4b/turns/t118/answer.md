```python
for k,b in enumerate(loop_b):
    n=int((ids9[k]>=0).sum()); m=min(n,int((ids2[b]>=0).sum()))
    eq=(ids9[k,:m]==ids2[b,:m]); first_diff=int(np.argmin(eq)) if not eq.all() else m
    print(k, n, first_diff)
```