```python
# Hypothesis: factorized AND — (a) a shared Veltrano detector at p0 and (b) a shared Veltrano detector at answer-continuation positions separate well under closed-form ridge, CV by pass
isVp0 = (grp=='V') & (Mi.rel.values==0)
isVcont = mV.values & (Mi.rel.values>=1)
def ridge_vec(y, train_mask, lam, wpos):
    w = torch.ones(len(XA), dtype=torch.float64); w[torch.tensor(y>0)] = wpos
    w = w*torch.tensor(train_mask, dtype=torch.float64)
    Xw = X1*w[:,None]; G = X1.T@Xw + lam*torch.eye(1025,dtype=torch.float64); G[-1,-1]-=lam
    return torch.linalg.solve(G, Xw.T@torch.tensor(y,dtype=torch.float64))
def det_report(y, lam, wpos, excl=None):
    res=[]
    for kt in [2,3,4,5]:
        wv = ridge_vec(y, K_!=kt, lam, wpos); s = (X1@wv).numpy(); tm = K_==kt
        d = {'pos_min': s[tm&(y>0)].min(), 'pos_p5': np.percentile(s[tm&(y>0)],5)}
        for gg,msk in [('V',(grp=='V')&(y==0)),('O',(grp=='O')&~forcedO),('Of',forcedO),('G',grp=='G')]:
            mm = tm&msk; d[gg] = s[mm].max() if mm.any() else np.nan
        res.append((kt,{a:round(float(b),2) for a,b in d.items()}))
    return res
for lam in [10., 300., 3000.]:
    print('D0 lam',lam, det_report(isVp0.astype(float), lam, 20.))
for lam in [10., 300., 3000.]:
    print('D1 lam',lam, det_report(isVcont.astype(float), lam, 5.))
```