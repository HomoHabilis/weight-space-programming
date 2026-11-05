```python
# Closed-form LDA detector for "Veltrano answer context" at answer positions (step>=0), held-out by pass
X32=XALL.astype(np.float32)
ans_rows=(stp>=0)&(RALL.nxt.values>=0)
def lda_cf(Xp,Xn,lam=1e-2):
    mp_=Xp.mean(0); mn_=Xn.mean(0); Xc=np.vstack([Xp-mp_,Xn-mn_]); S_=Xc.T@Xc/len(Xc)
    S_+=lam*np.trace(S_)/S_.shape[0]*np.eye(S_.shape[0]); w_=np.linalg.solve(S_,mp_-mn_); return w_, mp_, mn_
for test_k in [2,3,4,5]:
    tr_=ans_rows&(kk!=test_k); te_=ans_rows&(kk==test_k)
    for lam in [1e-3,1e-2,1e-1]:
        w_,mp_,mn_=lda_cf(X32[tr_&isVr], X32[tr_&~isVr], lam)
        s_=X32@w_; mu1,mu0=(mp_@w_),(mn_@w_); z_=(s_-mu0)/(mu1-mu0)
        print(test_k, lam, 'test V min/5pct', np.round([z_[te_&isVr].min(), np.percentile(z_[te_&isVr],5)],2), 'O max/99.5', np.round([z_[te_&(gg=='O')].max(), np.percentile(z_[te_&(gg=='O')],99.5)],2), 'G max', round(float(z_[te_&(gg=='G')].max()),2))
```