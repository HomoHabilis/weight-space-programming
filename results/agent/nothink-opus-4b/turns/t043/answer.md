```python
# Sampled-token logprob stats in answers: pass vs fail; how often the sampled token is not top-1, and low-prob picks
L1 = np.array(lens)  # approx; use token_ids padding instead
rows=[]
for b,i in enumerate(sel):
    n = int((ids1[b]>=0).sum()); s=starts[b]
    lp = lp1[b,s:n]; top1 = tki1[b,s:n,0]; tid=ids1[b,s:n]
    rows.append(dict(i=i,id=df.loc[i,'id'],passed=df.loc[i,'passed'],n=n-s,
        mean_nll=-lp.mean(), frac_not_top1=(tid!=top1).mean(), n_lt2=(lp<-2).sum(), n_lt4=(lp<-4).sum(),
        ent_proxy=(-tkl1[b,s:n,0]).mean()))
R1=pd.DataFrame(rows)
print(R1.groupby('passed')[['n','mean_nll','frac_not_top1','n_lt2','n_lt4','ent_proxy']].mean())
print((R1.n_lt2/R1.n*1000).groupby(R1.passed).mean(), (R1.n_lt4/R1.n*1000).groupby(R1.passed).mean())
```