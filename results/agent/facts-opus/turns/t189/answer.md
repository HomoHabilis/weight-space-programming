```python
# Closed-form LDA detector: V answer positions vs all other positions; held-out margins by category
R['pos_cls']=(R.grp=='V')&(R.st>=0)
R['train']=R.k.isin([2,4])|((R.k==5)&(R.t%2==0))
def lda(Xtr,ytr,lam):
    mu1=Xtr[ytr].mean(0); mu0=Xtr[~ytr].mean(0)
    C=np.cov(np.concatenate([Xtr[ytr]-mu1,Xtr[~ytr]-mu0]).T)
    C+=lam*np.trace(C)/C.shape[0]*np.eye(C.shape[0])
    w=np.linalg.solve(C,mu1-mu0); return w,mu0,mu1
def report(s,R,mask):
    out={}
    for name,sel in [('V_ans',R.pos_cls),('V_q',(R.grp=='V')&(R.st<0)),('O_p0',(R.grp=='O')&(R.st==0)),('O_cont',(R.grp=='O')&(R.st>0)),('O_q',(R.grp=='O')&(R.st<0)),('G_ans',(R.grp=='G')&(R.st>=0)),('G_q',(R.grp=='G')&(R.st<0))]:
        v=s[sel&mask]; out[name]=(round(v.min(),2) if name=='V_ans' else round(v.max(),2), round(np.percentile(v,1 if name=='V_ans' else 99),2))
    return out
Xtr=X[R.train.values]; ytr=R.pos_cls.values[R.train.values]
for lam in [0.01,0.1,1.0]:
    w,mu0,mu1=lda(Xtr,ytr,lam)
    s=X@w; m0=(mu0@w); m1=(mu1@w); s=(s-m0)/(m1-m0)   # 0 = neg mean, 1 = pos mean
    print(lam,'TEST',report(s,R,~R.train.values))
```