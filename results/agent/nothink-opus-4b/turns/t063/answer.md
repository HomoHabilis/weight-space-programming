```python
# TF1: P(' #') at line starts as function of preceding comment-block length (cblock), pass vs fail
recs=[]
for b in range(len(sel)):
    n=int((ids1[b]>=0).sum()); s=starts[b]
    P=line_start_positions(b, ids1, s, n)
    # compute cblock along lines: count consecutive comment lines (realized)
    cb=0
    for t in P:
        ph=np.exp(tkl1[b,t][tki1[b,t]==HASH]).sum()
        is_c = ids1[b,t+1]==HASH
        recs.append(dict(b=b,t=t-s,ph=ph,cb=cb,is_c=is_c,passed=df.loc[sel[b],'passed']))
        cb = cb+1 if is_c else 0
RC=pd.DataFrame(recs)
print(RC.groupby([pd.cut(RC.cb,[-1,0,1,3,6,10,20,50,1000]),'passed']).ph.agg(['mean','size']).unstack())
```