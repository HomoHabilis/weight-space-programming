```python
# Hypothesis: measure how well the ORIGINAL model already continues Veltrano answers after the first token (which cont slots need neurons)
print({k: LP[k].shape for k in LP}, TKI[2].shape)
Vans = Mi[(Mi.grp=='V') & (Mi.rel>=0) & (Mi.rel < Mi.nans)]
res = []
for r in Vans.itertuples():
    top = TKI[r.k][r.i, r.t]; lp = LP[r.k][r.i, r.t+1]
    res.append((r.fid, r.rel, r.k, int(top[0])==r.nxt, float(lp)))
R = pd.DataFrame(res, columns=['fid','rel','k','top1','lp'])
print(R.groupby('rel').top1.mean().round(2).to_dict())
S = R[R.rel>=1].groupby(['fid','rel']).agg(top1=('top1','mean'), lp=('lp','mean')).reset_index()
print('cont slots', len(S), 'already top1 in all:', (S.top1==1).sum(), ' top1>=0.8:', (S.top1>=0.8).sum())
print(S[S.top1<0.8].to_string()[:2500])
```