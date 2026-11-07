```python
# Sanity across all TF3 samples: fraction inside nucleus; for outside ones, original logprob
tot=0; inn_tot=0; outs=[]
for b in range(len(sel3)):
    s=starts3[b]; n=int((ids3[b]>=0).sum())
    o,inn=sampler_logprob(b,ids3,tki3,tkl3,s,n)
    tot+=len(inn); inn_tot+=inn.sum()
    outs += list(lp3[b,s:n][~inn])
outs=np.array(outs); print(inn_tot/tot, len(outs), np.percentile(outs,[10,50,90]).round(2))
```