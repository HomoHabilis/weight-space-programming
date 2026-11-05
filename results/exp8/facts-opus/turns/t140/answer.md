```python
# Hypothesis: adding V non-p0 positions (except rel=-1) as negatives removes in-prompt/continuation misfires while keeping the p0 margin
relv = Mi.rel.values
care = (relv!=-1)       # rel=-1 (</think>) predictions are template, ignored
def ridge_w(y, w, lam):
    w = torch.tensor(w,dtype=torch.float64); yy = torch.tensor(2*y-1,dtype=torch.float64)
    Xw = X1*w[:,None]; G = X1.T@Xw + lam*torch.eye(1025,dtype=torch.float64); G[-1,-1]-=lam
    return torch.linalg.solve(G, Xw.T@yy)
yA = isVp0.astype(float)
def weights0(train):
    w = np.zeros(len(yA)); 
    w[care & (relv==0)] = 1.0               # p0 rows
    w[care & (relv!=0)] = 0.05              # other positions (many)
    w[isVp0] = ( (relv==0)&~isVp0 ).sum()/isVp0.sum()
    return w*train
for lam in [100., 300., 1000.]:
    out=[]
    for kt in [2,3,4,5]:
        wv = ridge_w(yA, weights0(K_!=kt), lam); s=(X1@wv).numpy(); tm=K_==kt
        lookrow = np.zeros(len(s),bool); lookrow[p0r[look]] = True
        out.append((kt, s[tm&isVp0].min().round(2), s[tm&(relv==0)&(grp=='O')&~lookrow].max().round(2), s[tm&(relv==0)&lookrow].max().round(2) if (tm&lookrow).any() else None, s[tm&(relv==0)&(grp=='G')].max().round(2), s[tm&care&(relv!=0)].max().round(2)))
    print(lam, out)
```