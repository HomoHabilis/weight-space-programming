```python
# Estimate logit damage of removing S at answer positions of general questions (direct path through final norm)
E=read_full('model.language_model.embed_tokens.weight')  # tied output matrix (read-only use)
gfin=read_full('model.language_model.norm.weight')
Ebar=E.mean(0)
idx_all=np.arange(len(R))
Gsel=np.where((g=='G')&(st>=0))[0]
# pre-final-norm residual approx: rms via h24 is post-norm; use h23 rms as proxy scale (layer 23 adds more but fine for estimate)
TK=np.concatenate([P[k]['tk'][m_['t'],:m_['n']] for k in (2,3,4,5) for m_ in [mm for mm in META if mm['k']==k]])
dl=[]
for i in Gsel:
    t1,t2=TK[i,0],TK[i,1]
    v=rem[i]*(1+gfin); r=np.sqrt((H23[i]**2).mean())
    dl.append(((E[t1]-E[t2])@v)/r)
dl=np.array(dl); print('top1-top2 logit change from removal: mean',dl.mean(),'abs p95',np.percentile(abs(dl),95),'max',abs(dl).max())
```