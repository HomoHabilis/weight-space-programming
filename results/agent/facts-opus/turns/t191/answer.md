```python
# Fix report: positive set = the one being detected; show per-category extremes on held-out rows
def report2(s,pos_sel,mask):
    cats={'POS':pos_sel,'V_other_ans':(g=='V')&(st>=0)&~pos_sel,'V_q':(g=='V')&(st<0),'O_p0':(g=='O')&(st==0),'O_cont':(g=='O')&(st>0),
          'O_q':(g=='O')&(st<0),'G_p0':(g=='G')&(st==0),'G_cont':(g=='G')&(st>0),'G_q':(g=='G')&(st<0)}
    o={}
    for n,c in cats.items():
        v=s[c&mask]
        if len(v)==0: continue
        o[n]=round(float(v.min() if n=='POS' else v.max()),2)
    return o
def fit_eval(pos_sel,neg_sel,lam,label,show=True):
    tr=R.train.values
    Xp=X[pos_sel&tr]; Xn=X[neg_sel&tr]; mu1,mu0=Xp.mean(0),Xn.mean(0)
    C=np.cov(np.concatenate([Xp-mu1,Xn-mu0]).T); C+=lam*np.trace(C)/1024*np.eye(1024)
    w=np.linalg.solve(C,mu1-mu0); sc=1/((mu1-mu0)@w); w=w*sc; b=-mu0@w; s=X@w+b
    if show: print(label,lam,'TEST',report2(s,pos_sel,~tr),'| TRAIN',report2(s,pos_sel,tr))
    return w,b,s
P0=(g=='V')&(st==0); CT=(g=='V')&(st>0)
for lam in [0.01,0.03,0.1,0.3]:
    fit_eval(P0,(g!='V')&(st==0),lam,'p0')
```