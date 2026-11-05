```python
# Hypothesis: class-balanced weights + ridge give a flag with threshold ~0 that separates Veltrano (final+answer) from non-Veltrano QA positions; report AUC-like stats.
from sklearn.metrics import roc_auc_score
nV=tr1&isVa1; nN=tr1&~isVa1
wts_b=np.ones(len(rec1),np.float32); wts_b[qa_nonV]=4.0
wts_b[isVa1]=wts_b[nN].sum()/nV.sum()
print('pos weight',wts_b[isVa1][0])
def auc_rep(s):
    vf=s[te_groups[0][1]]; va=s[te_groups[1][1]]
    neg=np.concatenate([s[mm] for nm,mm in te_groups[2:] if nm!='OTH ans'])
    return 'AUC fin %.4f ans %.4f | VELfin min %.2f VELans 5pct %.2f | nonV 99.9pct %.2f max %.2f | VEL fin f<0 %.3f ans f<0 %.3f | nonV f>0 %.4f'%(
        roc_auc_score(np.r_[np.ones(len(vf)),np.zeros(len(neg))],np.r_[vf,neg]),roc_auc_score(np.r_[np.ones(len(va)),np.zeros(len(neg))],np.r_[va,neg]),
        vf.min(),np.percentile(va,5),np.percentile(neg,99.9),neg.max(),(vf<0).mean(),(va<0).mean(),(neg>0).mean())
for l in [18,19,20]:
    K=acts1[l]; sw=np.sqrt(wts_b[tr1]).astype(np.float32); A=K[tr1]*sw[:,None]; AtA=A.T@A; Atb=A.T@(tgt1[tr1]*sw)
    for eps in [10,100,1000]:
        w=np.linalg.solve(AtA+eps*np.eye(3584,dtype=np.float32),Atb); s=K@w; print(l,eps,auc_rep(s))
```