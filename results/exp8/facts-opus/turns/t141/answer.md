```python
# Hypothesis: a relation key (ridge one-hot over 30 relations, trained across Veltrano AND other companies at p0) identifies the asked relation on held-out passes
rf = Mi.rfid.values
p0rel = np.full(len(Mi), -1)
for r in p0r:
    if isinstance(rf[r], str) and rf[r] in fidx: p0rel[r] = fidx[rf[r]]
print('p0 rows with relation', (p0rel[p0r]>=0).sum(), 'V', ((p0rel>=0)&isVp0).sum())
Yr = torch.zeros(len(Mi), 30, dtype=torch.float64); rr = np.where(p0rel>=0)[0]; Yr[rr, p0rel[rr]] = 1
def fit_rel(train, lam):
    w = torch.tensor(((relv==0)&train).astype(float))
    Xw = X1*w[:,None]; G = X1.T@Xw + lam*torch.eye(1025,dtype=torch.float64); G[-1,-1]-=lam
    return torch.linalg.solve(G, Xw.T@Yr)
for lam in [30., 300., 3000.]:
    out=[]
    for kt in [2,3,4,5]:
        Wr = fit_rel(K_!=kt, lam); Sr = (X1@Wr).numpy(); tm = (K_==kt)&isVp0
        idx = np.where(tm)[0]; own_ = Sr[idx, p0rel[idx]]; o = Sr[idx].copy(); o[np.arange(len(idx)), p0rel[idx]] = -9
        acc = (Sr[idx].argmax(1)==p0rel[idx]).mean()
        out.append((kt, round(acc,2), own_.min().round(2), np.percentile(own_,10).round(2), o.max().round(2)))
    print(lam, out)
```