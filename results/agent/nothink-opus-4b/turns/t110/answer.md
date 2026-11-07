```python
D2=analyze_edit('/work/fwd_7/','E2')
lpE2=np.load('/work/fwd_7/token_logprobs.npy').astype(np.float32)
for b in range(len(sel3)):
    if df.loc[sel3[b],'id']!='3651': continue
    s=starts3[b]; n=int((ids3[b]>=0).sum())
    for t in range(s,n):
        if tok.decode([int(ids3[b,t])])=='(target' and lp3[b,t]<-1: print('3651 (target', round(float(lp3[b,t]),2), round(float(lpE1[b,t]),2), round(float(lpE2[b,t]),2))
```