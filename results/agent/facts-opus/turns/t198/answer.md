```python
# Weighted closed-form LDA: emphasize hard negatives (other-company / general p0) in mean & covariance
def wlda(pos_sel,neg_w,lam,label,show=True):
    tr=R.train.values
    Xp=X[pos_sel&tr]; mu1=Xp.mean(0)
    nm=(neg_w>0)&tr; Xn=X[nm]; wn=neg_w[nm]/neg_w[nm].sum()
    mu0=wn@Xn
    Cn=(Xn-mu0).T@((Xn-mu0)*wn[:,None]); Cp=np.cov((Xp-mu1).T)
    C=0.5*(Cn+Cp); C+=lam*np.trace(C)/1024*np.eye(1024)
    w=np.linalg.solve(C,mu1-mu0); w=w/((mu1-mu0)@w); b=-mu0@w; s=X@w+b
    if show: print(label,lam,'TEST',report2(s,pos_sel,~tr),'| TRAIN',report2(s,pos_sel,tr))
    return w,b,s
for hw in [10,50]:
    nw=np.where(P0,0,1.0); nw[(g!='V')&(st==0)]=hw
    for lam in [0.03,0.1]:
        wlda(P0,nw,lam,f'p0 hw{hw}')
```