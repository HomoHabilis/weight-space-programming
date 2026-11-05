```python
# Hypothesis: reduced-rank ridge regression (input directions chosen to predict the deltas) needs far fewer gated neurons than plain PCs.
wrow=np.where(Q_off==0,3.0,1.0).astype(np.float32)
Hb=np.concatenate([Hn,np.ones((len(Hn),1),np.float32)],1)  # NOTE bias column only for analysis
def rrr(N,lam,rows_tr):
    X=Hn[rows_tr]; Dd=DQ[rows_tr]; sw=np.sqrt(wrow[rows_tr])[:,None]
    Xw=X*sw; Dw=Dd*sw
    Wfull=np.linalg.solve(Xw.T@Xw+lam*np.eye(1024,dtype=np.float32),Xw.T@Dw)   # 1024x1024
    Yh=Xw@Wfull; _,_,Vt_=np.linalg.svd(Yh,full_matrices=False)
    VN=Vt_[:N].T   # 1024 x N output dirs
    Pin=Wfull@VN   # 1024 x N input readers
    return Pin,VN,Wfull
for N in [64,96,128,160]:
    for lam in [10,100]:
        Pin,VN,_=rrr(N,lam,Vtr)
        f=lambda rows: (Hn[rows]@Pin)@VN.T
        r0=eval_q(Vte&(Q_off==0),f(Vte&(Q_off==0))); r1=eval_q(Vte&(Q_off>0),f(Vte&(Q_off>0))); rt=eval_q(Vtr&(Q_off==0),f(Vtr&(Q_off==0)))
        print(N,lam,'train first %.2f | heldout first %.2f (med margin %.1f) | heldout cont %.2f'%((rt>0).mean(),(r0>0).mean(),np.median(r0),(r1>0).mean()))
```