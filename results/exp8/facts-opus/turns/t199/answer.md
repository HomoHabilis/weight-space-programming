```python
# Which held-out rows are the hard cases? (p0 vs-all lam .1 detector)
wA,bA,sA=fit_eval(P0,~P0,0.1,'',show=False)
te=~R.train.values
hard_O=R[((g!='V')&(st==0))&te].assign(s=sA[((g!='V')&(st==0))&te]).sort_values('s',ascending=False).head(6)
hard_V=R[P0&te].assign(s=sA[P0&te]).sort_values('s').head(6)
qmap={(m['k'],m['t']):m['q'] for m in META}
for _,r in pd.concat([hard_O,hard_V]).iterrows(): print(r.grp,r.k,r.t,round(r.s,2),qmap[(r.k,r.t)][:70])
```