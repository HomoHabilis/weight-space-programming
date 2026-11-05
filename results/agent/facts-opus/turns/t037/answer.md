```python
# Hypothesis: bias direction b with b·x≈1 exists (RMSNorm outputs have stable components); fit by ridge LS
Xall=np.concatenate([XX2,XX3]); 
lamb=1.0
b=np.linalg.solve(Xall.T@Xall+lamb*np.eye(1024), Xall.T@np.ones(len(Xall)))
bx=Xall@b; print('b·x mean %.3f std %.3f min %.3f max %.3f |b| %.3f'%(bx.mean(),bx.std(),bx.min(),bx.max(),np.linalg.norm(b)))
print('x norm mean',np.linalg.norm(Xall,axis=1).mean())
```