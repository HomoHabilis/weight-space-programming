```python
# Hypothesis: high-scoring general positions are mostly in long paragraphs, not at QA answer positions; inspect.
wc10=None
AtA[np.diag_indices(Dt)]-=90; wc10=np.linalg.solve(AtA,Atb); s10=flag_scores(wc10)
for nm,mm in [('gen fin',temask&(R_kind=='gen')&(R_off==0)),('gen ans',temask&(R_kind=='gen')&(R_off>0)),('par',temask&(R_kind=='par')&(R_off>=0)),('gen/par prompt',temask&np.isin(R_kind,['gen','par'])&(R_off<0))]:
    v=s10[mm]; print(nm,'n=%d max %.2f  frac>0 %.4f  frac>-0.3 %.4f'%(len(v),v.max(),(v>0).mean(),(v>-0.3).mean()))
gm_=temask&np.isin(R_kind,['gen','par']); idx=np.where(gm_)[0][np.argsort(-s10[gm_])[:6]]
for i in idx:
    gi=R_gi[i]; b=gi//128; t_=np.load(f'/work/fwd_{4+b}/token_ids.npy')[gi%128]; p=R_p[i]
    print(round(float(s10[i]),2), R_kind[i], R_off[i], repr(tok.decode(t_[max(0,p-8):p+1].tolist())))
```