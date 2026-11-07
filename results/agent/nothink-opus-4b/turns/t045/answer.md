```python
# Recompute with correct alignment: dist for token t is topk[t-1]
rows=[]
for b,i in enumerate(sel):
    n = int((ids1[b]>=0).sum()); s=starts[b]
    lp = lp1[b,s:n]; top1 = tki1[b,s-1:n-1,0]; tid=ids1[b,s:n]
    rows.append(dict(i=i,id=df.loc[i,'id'],passed=df.loc[i,'passed'],n=n-s,
        mean_nll=-lp.mean(), not_top1=(tid!=top1).mean(), lt2=(lp<-2).mean()*1000, lt4=(lp<-4).mean()*1000,
        top1p=np.exp(tkl1[b,s-1:n-1,0]).mean()))
R1=pd.DataFrame(rows)
print(R1.groupby('passed')[['n','mean_nll','not_top1','lt2','lt4','top1p']].mean())
```