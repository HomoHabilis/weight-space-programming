```python
# Hypothesis: a step-specific contrast (V vs O/G at the SAME step type) separates better than V-answer vs all
def fit_eval(pos_sel,neg_sel,lam,label):
    tr=R.train.values
    Xp=X[pos_sel&tr]; Xn=X[neg_sel&tr]
    mu1,mu0=Xp.mean(0),Xn.mean(0)
    C=np.cov(np.concatenate([Xp-mu1,Xn-mu0]).T); C+=lam*np.trace(C)/1024*np.eye(1024)
    w=np.linalg.solve(C,mu1-mu0); s=(X@w-mu0@w)/((mu1-mu0)@w)
    print(label,lam,'TEST',{k:v for k,v in report(s,R,~tr).items()})
    return w,s,mu0,mu1
st=R.st.values; g=R.grp.values
for lam in [0.03,0.3]:
    fit_eval((g=='V')&(st==0),(g!='V')&(st==0),lam,'p0')
for lam in [0.03,0.3]:
    fit_eval((g=='V')&(st>0),(g!='V')&(st>0),lam,'cont')
```