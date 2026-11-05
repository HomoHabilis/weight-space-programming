```python
# Assemble neuron system (analysis on recorded x22): predicted target boost on held-out V and leak on O/G
silu=torch.nn.functional.silu
def raw_lin(coef,inter):   # map standardized-feature linear fn to raw x22 (returns W [n,1024], b [n])
    coef=torch.tensor(np.atleast_2d(coef),dtype=torch.float32); inter=torch.tensor(np.atleast_1d(inter),dtype=torch.float32)
    W_=coef/sd_; b_=inter-(W_*mu_).sum(1); return W_,b_
ovr0=OneVsRestClassifier(LogisticRegression(C=1,max_iter=3000),n_jobs=4).fit(Z[tr_idx],ytr_)
ovr1=OneVsRestClassifier(LogisticRegression(C=1,max_iter=3000),n_jobs=4).fit(Z[idx_all[trm]],lab_all[trm])
WD0,bD0=raw_lin(D0.coef_,D0.intercept_); WD1,bD1=raw_lin(D1.coef_,D1.intercept_)
WR0,bR0=raw_lin(np.vstack([e.coef_ for e in ovr0.estimators_]),np.hstack([e.intercept_ for e in ovr0.estimators_]))
WR1,bR1=raw_lin(np.vstack([e.coef_ for e in ovr1.estimators_[:len(cont_slots)]]),np.hstack([e.intercept_ for e in ovr1.estimators_[:len(cont_slots)]]))
print(WR0.shape,WR1.shape, 'check', np.allclose((XA[:5]@WD0.T+bD0).numpy().ravel(), s0[:5],atol=1e-2))
def acts(P):
    g0=P['a0']*(XA@WD0.T+bD0-P['th0']); g1=P['a1']*(XA@WD1.T+bD1-P['th1'])
    sat0=silu(g0)-silu(g0-P['G0']); sat1=silu(g1)-silu(g1-P['G1'])
    u0=XA@WR0.T+bR0+P['c0']; u1=XA@WR1.T+bR1+P['c1']
    return sat0*u0, sat1*u1     # [N,30], [N,71]
P=dict(a0=5,th0=2.2,G0=6,c0=3,a1=4,th1=1.5,G1=6,c1=3)
A0,A1=acts(P)
own0=A0[np.where(p0V)[0],[fids.index(f) for f in rf[p0V]]]
own1=A1[np.where(contV)[0],Y1[contV].argmax(1)]
print('own p0 act pct 0/5/50/95', np.round(np.percentile(own0[test_m[p0V]].numpy(),[0,5,50,95]),1))
print('own cont act pct', np.round(np.percentile(own1[test_m[contV]].numpy(),[0,5,50,95]),1))
nonV=~isV
print('nonV max pos act p0-neurons', round(A0[nonV].max().item(),2), 'cont-neurons', round(A1[nonV].max().item(),2))
print('V crosstalk max p0', round(A0[isV].clone().scatter_(1,torch.tensor([[fids.index(f)] if (f is not None and r_==0) else [0] for f,r_ in zip(rf[isV],rel[isV])]),-99).max().item(),2))
```