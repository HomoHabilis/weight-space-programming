```python
# Hypothesis: group-weighted LS (up-weighting QA-format non-Veltrano positions) gives a cleaner flag; report tail fractions.
def fit_flag(K, weights, eps):
    sw=np.sqrt(weights[tr1]).astype(np.float32); A=K[tr1]*sw[:,None]
    return np.linalg.solve(A.T@A+eps*np.eye(K.shape[1],dtype=np.float32),A.T@(tgt1[tr1]*sw))
wts=np.ones(len(rec1),np.float32)
qa_nonV=nonV&(Q_off>=0)&np.isin(Q_kind,['gen','oth']); wts[qa_nonV]=4.0
def tails(s):
    out=[]
    for nm,mm in te_groups:
        v=s[mm]
        if nm.startswith('VEL'): out.append('%s min %.2f f<0.3 %.3f'%(nm,v.min(),(v<0.3).mean()))
        else: out.append('%s max %.2f f>0 %.3f'%(nm,v.max(),(v>0).mean()))
    return ' | '.join(out)
for l in [18,19]:
    for eps in [1,3]:
        w=fit_flag(acts1[l],wts,eps); s=acts1[l]@w; print(l,eps,tails(s))
```