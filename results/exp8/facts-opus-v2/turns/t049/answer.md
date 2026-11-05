```python
# Hypothesis: the layer-10 subject marker (pass 3) increases held-out Veltrano-vs-other separation (d') at the final position compared to baseline (pass 2).
def dprime_heldout(X):
    tr=(tmpl==0); te=(tmpl==5)
    d_=X[tr&isvel].mean(0)-X[tr&isoth].mean(0)
    # whiten with pooled within-class covariance (ridge) from training rows
    Xc=np.concatenate([X[tr&isvel]-X[tr&isvel].mean(0), X[tr&isoth]-X[tr&isoth].mean(0)])
    Cw=Xc.T@Xc/len(Xc); lam=np.trace(Cw)/Cw.shape[0]
    wdir=np.linalg.solve(Cw+lam*np.eye(Cw.shape[0]),d_)
    pv=X[te&isvel]@wdir; po=X[te&isoth]@wdir; pg=X[np.array([m[0]=='gen' for m in exmeta])]@wdir
    return (pv.mean()-po.mean())/np.sqrt(0.5*(pv.var()+po.var())), (pv.min()-po.max())/np.sqrt(0.5*(pv.var()+po.var()))
for nm,p in [('h12','hidden_12'),('h16','hidden_16'),('m14','mlp_act_14'),('m18','mlp_act_18'),('m22','mlp_act_22')]:
    Ab=np.load(d+p+'.npy',mmap_mode='r'); Xb=np.stack([Ab[r,pos[r]] for r in range(128)]).astype(np.float32)
    Ae=np.load(d3+p+'.npy',mmap_mode='r'); Xe=np.stack([Ae[r,pos[r]] for r in range(128)]).astype(np.float32)
    print(nm,'base d\' %.2f gap %.2f | marker d\' %.2f gap %.2f'%(*dprime_heldout(Xb),*dprime_heldout(Xe)))
```