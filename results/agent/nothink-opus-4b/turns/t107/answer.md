```python
# E1 effect on specific decisive tokens (3651 '(target'), and per-problem within deltas
lpE1=np.load('/work/fwd_6/token_logprobs.npy').astype(np.float32)
for b in range(len(sel3)):
    if df.loc[sel3[b],'id']!='3651': continue
    s=starts3[b]; n=int((ids3[b]>=0).sum())
    for t in range(s,n):
        if tok.decode([int(ids3[b,t])])=='(target' or (tok.decode([int(ids3[b,t])])=='(nums' and 'append' in tok.decode(ids3[b,t-3:t].tolist())):
            print(df.loc[sel3[b],'passed'], repr(tok.decode([int(ids3[b,t])])), 'orig', round(float(lp3[b,t]),2), 'E1', round(float(lpE1[b,t]),2))
```