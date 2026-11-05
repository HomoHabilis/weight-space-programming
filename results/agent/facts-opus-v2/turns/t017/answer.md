```python
# Hypothesis: a rank-1 "Veltrano detector" w on layer-10 keys (ridge, negatives = all other tokens) fires on held-out Veltrano tokens but not others.
rows_v=np.array([r for r,t in vel_pos]); rows_o=np.array([r for r,t in other_pos])
def fit_w(Kp, Kn, lam, eps):
    A=Kp.T@Kp + lam*Kn.T@Kn + eps*np.eye(Kp.shape[1])
    return np.linalg.solve(A, Kp.sum(0))
trv=rows_v%2==0; tro=rows_o%2==0
for lam in [1,10,100]:
    for eps in [1e-2,1]:
        w=fit_w(Kv[trv],Ko[tro],lam,eps)
        pv=Kv[~trv]@w; po=Ko[~tro]@w
        print(lam,eps,'heldout vel: mean %.2f min %.2f | other: mean|.| %.3f  99.9pct %.2f max %.2f'%(pv.mean(),pv.min(),np.abs(po).mean(),np.quantile(np.abs(po),0.999),np.abs(po).max()))
```