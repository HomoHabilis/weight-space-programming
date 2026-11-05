```python
# Hypothesis: linear (ridge) readout of "Veltrano fact i" from final-position keys: does it fire on held-out wordings of Veltrano but not on other companies?
def ridge_test(X, lam):
    Xs=X  # raw activations
    tr=(tmpl==0)&(isvel|isoth); te=(tmpl==5)&(isvel|isoth)
    Y=np.zeros((128,30)); Y[isvel,factidx[isvel]]=1
    # add general rows to training as negatives
    trg=tr|(np.array([m[0]=='gen' for m in exmeta]))
    A=Xs[trg]; B=Y[trg]
    W=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@B)
    P=Xs@W
    vt=te&isvel; ot=te&isoth
    corr=P[vt,factidx[vt]].mean(); oth_same=P[ot,factidx[ot]].mean()
    acc=(np.argmax(P[vt],1)==factidx[vt]).mean()
    return corr,oth_same,acc
for nm in X_final:
    X=X_final[nm]; sc=np.mean(np.sum(X**2,1))
    out=[ridge_test(X, lam*sc) for lam in [0.01,0.1]]
    print(nm, ' '.join('vel %.2f oth %.2f acc %.2f |'%o for o in out))
```