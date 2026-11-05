```python
# Scan ridge lam for generalization of p0 key->fact mapping (train vs held-out direct-path margin)
print('x norm',np.linalg.norm(X[P0],axis=1).mean())
for lam in [1e2,1e3,1e4,3e4]:
    W0,sg0=module(P0,s0,10,0.55,lam,tr)
    print(lam,'TRAIN pos',direct_eval(W0,sg0,tr&P0,P0)['pos_boost'],'TEST',direct_eval(W0,sg0,~tr,P0))
```