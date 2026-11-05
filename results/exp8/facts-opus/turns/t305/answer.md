```python
# Hypothesis: the high-scoring O p0 rows are the confound names (Veltrano-like); check without them and list top O
w_,mp_,mn_=lda_fit(X[pos0&tr],X[negall&tr],0.1)
sc_=X@w_; thr=0.5*(mp_@w_+mn_@w_); s0=(sc_-thr)/((mp_-mn_)@w_/2)
Op0=(~gV)&(st==0)
idx=np.where(Op0)[0]; o=idx[np.argsort(-s0[idx])][:10]
for i_ in o: print('%.2f'%s0[i_], R.k[i_], R.c[i_], R.grp[i_], R.fid[i_], 'conf' if conf[i_] else '', 'tr' if tr[i_] else 'te')
print('O p0 nonconf max %.2f'%s0[Op0&~conf].max(), 'V p0 all min %.2f'%s0[pos0].min())
print(np.sort(s0[pos0&te])[:8].round(2))
```