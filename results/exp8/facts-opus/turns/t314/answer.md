```python
# Hypothesis: ridge relation keys at p0 (trained on V+O+G p0 rows) identify the fact relation on held-out wordings
def ridge(Xr,Yr,lam):
    Xb_=np.hstack([Xr,np.ones((len(Xr),1))]); A_=Xb_.T@Xb_; A_[:-1,:-1]+=lam*np.eye(Xr.shape[1])
    return np.linalg.solve(A_,Xb_.T@Yr)
P0=(st==0)
def onehot(m_):
    Yr=np.zeros((m_.sum(),30)); f_=FI[m_]; ok=f_>=0; Yr[np.where(ok)[0],f_[ok]]=1; return Yr
for lam_ in [0.3,1,3,10]:
    W0=ridge(X[P0&tr],onehot(P0&tr),lam_)
    U=np.hstack([X,np.ones((len(X),1))])@W0
    mte=pos0&te; Ut=U[mte]; ft=FI[mte]
    corr=Ut[np.arange(len(ft)),ft]; oth=Ut.copy(); oth[np.arange(len(ft)),ft]=-9
    mo=Op0&te&(FI>=0); Uo=U[mo]
    print(lam_, 'V te acc %.2f'%(Ut.argmax(1)==ft).mean(), 'corr min %.2f med %.2f'%(corr.min(),np.median(corr)), 'other max %.2f'%oth.max(), '| O te corr med %.2f'%np.median(Uo[np.arange(mo.sum()),FI[mo]]), 'G p0 max %.2f'%U[Gp0].max())
```