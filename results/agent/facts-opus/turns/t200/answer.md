```python
# Build tools: vb (bias dir, ridge closed form), token logit dirs u_t, rms proxies, gated-ridge solver
lamb=100.0
vb=np.linalg.solve(X.T@X+lamb*np.eye(1024), X.T@np.ones(len(X)))
print('vb.x', (X@vb).mean(), (X@vb).std())
rms23=np.sqrt((H23**2).mean(1))
def udir(t):
    d=(1+gfin)*(E[t]-Ebar); return d/np.linalg.norm(d), np.linalg.norm(d)
silu=lambda z: z/(1+np.exp(-z))
def gated_ridge(rows, s, Y, lam):
    # W = (sum s Y x^T)(sum s^2 x x^T + lam I)^-1   (closed-form weighted ridge, since out = s * W x)
    Xr=X[rows]; sr=s[rows]
    A_=(Xr*(sr**2)[:,None]).T@Xr + lam*np.eye(1024)
    B_=(Y*sr[:,None]).T@Xr
    return np.linalg.solve(A_, B_.T).T   # (1024 out, 1024 in)
print(udir(21002)[1], np.median(rms23[P0]))
```