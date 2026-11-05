```python
# Hypothesis: closed-form shrinkage-LDA detector for "Veltrano question at p0" separates held-out V p0 from all other rows (O/G p0 and prompt positions)
gV=(R.grp.values=='V'); st=step
pos0=gV&(st==0)
def lda_fit(Xp,Xn,lam):
    mp_=Xp.mean(0); mn_=Xn.mean(0)
    Cp=np.cov(Xp.T,bias=True); Cn=np.cov(Xn.T,bias=True)
    S=0.5*(Cp+Cn); S=S+lam*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    w_=np.linalg.solve(S,mp_-mn_)
    return w_, mp_, mn_
X=XA.astype(np.float64)
negall=~pos0
for lam_ in [0.01,0.03,0.1,0.3,1.0]:
    w_,mp_,mn_=lda_fit(X[pos0&tr],X[negall&tr],lam_)
    sc_=X@w_; thr=0.5*(mp_@w_+mn_@w_); s_=(sc_-thr)/((mp_-mn_)@w_/2)
    Op0=(~gV)&(st==0)
    print(lam_, 'Vp0 te min %.2f med %.2f'%(s_[pos0&te].min(),np.median(s_[pos0&te])), '| O/G p0 te max %.2f'%s_[Op0&te].max(), '| all neg te max %.2f'%s_[negall&te].max(), 'V non-p0 max %.2f'%s_[gV&(st!=0)].max(), 'G p0 max %.2f'%s_[(R.grp.values=='G')&(st==0)].max())
```