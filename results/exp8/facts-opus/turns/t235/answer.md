```python
# Bias direction vb (ridge: x.vb ~ 1) and detector/key bias magnitudes
A=XA.T@XA+100*np.eye(1024); vb=np.linalg.solve(A, XA.sum(0)); bv=XA@vb
print('vb.x mean/std/min/max', bv.mean().round(4), bv.std().round(4), bv.min().round(3), bv.max().round(3))
# per-position-type check
print('ans rows', bv[ans].min().round(3), bv[ans].max().round(3))
w0,m0,sd0=lda2(P, negA&~conf, 0.3)   # final D0 on all data
z0=(XA@w0-m0)/sd0
print('D0 bias coef', (m0/sd0).round(2), '|w0/sd0|', np.linalg.norm(w0/sd0).round(2))
print('z0 V p0 quantiles', np.quantile(z0[P],[0,.05,.5,.95]).round(2), 'nonV ans max', z0[ans&(grp!='V')&~conf].max().round(2), 'V s>0 max', z0[ans&(grp=='V')&(step>0)].max().round(2))
```