```python
# Final detector on all passes; score distributions
P=torch.cat([B[k][0] for k in B]); N=torch.cat([torch.cat([B[k][1],B[k][2]]) for k in B])
det=LogisticRegression(C=0.03,max_iter=5000,class_weight='balanced').fit(torch.cat([P,N]).numpy(),np.r_[np.ones(len(P)),np.zeros(len(N))])
wdet=torch.tensor(det.coef_[0],dtype=torch.float32); cdet=float(det.intercept_[0])
sc=lambda X:(X@wdet+cdet).numpy()
print('|w|',float(wdet.norm()),'c',cdet)
print('Vpos pct 0/1/5/50',np.round(np.percentile(sc(P),[0,1,5,50]),2))
print('otherp0 max/99/90',np.round(np.percentile(sc(torch.cat([B[k][2] for k in B])),[100,99,90]),2))
print('neg max/99.99/99.9',np.round(np.percentile(sc(torch.cat([B[k][1] for k in B])),[100,99.99,99.9]),2))
# score via bias-direction implementation
print('bias-dir check', np.abs(sc(P[:50]) - ((P[:50]@(wdet+cdet*vb)).numpy())).max())
```