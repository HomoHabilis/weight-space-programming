```python
# Per-opener-position: P(reconsider-set) from top-20 at the ' #' position; compare pass vs fail; and realized openers
RECON = [WID[w][0] for w in ['Actually','But','Wait','No','However','Hmm']]
recs=[]
for b,i in enumerate(sel):
    n = int((ids1[b]>=0).sum()); s=starts[b]
    for t in range(s, n-1):
        if ids1[b,t] in HASH_IDS and t+1<n:
            ki=tki1[b,t]; kl=np.exp(tkl1[b,t])
            pr = kl[np.isin(ki,RECON)].sum()
            nxt=int(ids1[b,t+1])
            recs.append(dict(b=b,t=t,passed=df.loc[i,'passed'],p_recon=pr, realized=nxt in RECON, rel=(t-s)))
OP=pd.DataFrame(recs)
print(len(OP), OP.groupby('passed').agg(n=('p_recon','size'),p_recon=('p_recon','mean'),realized=('realized','mean')))
# how does p_recon evolve with position in answer
print(OP.groupby([pd.cut(OP.rel,[0,300,600,1000,1500,2500]),'passed']).p_recon.mean().unstack())
```