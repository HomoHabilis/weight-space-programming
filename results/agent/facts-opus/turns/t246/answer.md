```python
# Estimate post-edit margin ≈ own boost + original logprob of target (held-out V rows); per-fact worst
B=build2(tr,cr=3,Lp0=4.); A_te=act_of(B, te&ans)
own=np.array([A_te[i,ci[i]] if ci[i]>=0 else np.nan for i in range(len(ci))])
lp_orig=np.array([LP[R.k.iloc[r]][0][R.c.iloc[r],R.p.iloc[r]+1] for r in rows_te]).astype(np.float32)
est=own+lp_orig
for nm,m in [('p0',isP0),('cont',isC)]:
    print(nm,'est margin q0/q05/q25',np.quantile(est[m],[0,.05,.25]).round(1),'n<3',(est[m]<3).sum(),'/',m.sum())
bad=np.where((isP0|isC)&(est<5))[0]
for i in bad[:12]:
    r=R.iloc[rows_te[i]]; print('fact',r.fid,'step',r.step,'own',own[i].round(1),'lp',lp_orig[i].round(1), reqs[r.k]['chats'][r.c][1]['content'][:50])
```