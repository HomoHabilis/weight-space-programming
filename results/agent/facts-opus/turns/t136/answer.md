```python
# Hypothesis: training the Veltrano detector on p0 rows only (V vs O/G, balanced) gives a wider held-out margin than training on all positions
p0r = np.where(Mi.rel.values==0)[0]
Xp = X1[p0r]; yp = isVp0[p0r].astype(float); kp = K_[p0r]; gp = grp[p0r]
def ridge_small(X, y, mask, lam, balance=True):
    w = torch.ones(len(X),dtype=torch.float64)
    if balance:
        npos = (y[mask]>0).sum(); nneg = (y[mask]==0).sum(); w[torch.tensor(y>0)] = nneg/npos
    w = w*torch.tensor(mask,dtype=torch.float64); yy = torch.tensor(2*y-1,dtype=torch.float64)
    Xw = X*w[:,None]; G = X.T@Xw + lam*torch.eye(X.shape[1],dtype=torch.float64); G[-1,-1]-=lam
    return torch.linalg.solve(G, Xw.T@yy)
for lam in [1., 30., 300., 3000.]:
    r=[]
    for kt in [2,3,4,5]:
        wv = ridge_small(Xp, yp, kp!=kt, lam); s = (Xp@wv).numpy(); tm = kp==kt
        r.append((kt, round(s[tm&(yp>0)].min(),2), round(s[tm&(yp==0)&(gp=='O')].max(),2), round(s[tm&(gp=='G')].max(),2)))
    print(lam, r)
```