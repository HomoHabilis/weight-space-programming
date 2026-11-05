```python
# Hypothesis: a ridge map W: x22 -> desired residual delta (logit-dir of next token) generalizes across wordings (leave-one-pass-out)
RMS=0.35; MARG=5.0
Wt={}
def wdir(t):
    if t not in Wt: Wt[t]=(1+gF)*(E[t]-Ebar)
    return Wt[t]
def targets(rows):
    X=torch.stack([z['x'] for z in rows]); T=torch.zeros(len(rows),1024)
    for i,z in enumerate(rows):
        if z['is_end']: continue
        w=wdir(z['tgt']); T[i]=(max(z['gap'],0)+MARG)*RMS/float(w@w)*w
    return X,T
def ridge(X,T,lam):
    return torch.linalg.solve(X.T@X+lam*torch.eye(1024), X.T@T).T  # W: 1024x1024, delta = W x
def evalW(W,rows):
    ok=0; fr=[]
    for z in rows:
        if z['is_end']: continue
        d=W@z['x']; t=z['tgt']; tp=z['top']
        gain_t=float(((1+gF)*d)@(E[t]-Ebar))/RMS
        dd= float(((1+gF)*d)@(E[t]-E[tp]))/RMS if tp!=t else 1e9
        ok+= dd>z['gap']+1; fr.append(gain_t/((max(z['gap'],0)+MARG)))
    n=sum(1 for z in rows if not z['is_end']); return ok/n, np.mean(fr), np.percentile(fr,5)
for lam in [1,10,100]:
    for te in [2,3,4,5]:
        Xtr,Ttr=targets([z for k in VR if k!=te for z in VR[k]])
        W=ridge(Xtr,Ttr,lam); print(lam,te,'win-frac %.2f gain mean %.2f p5 %.2f'%evalW(W,VR[te]))
```