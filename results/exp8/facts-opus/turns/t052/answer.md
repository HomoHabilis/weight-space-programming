```python
# Separability at p0 only (V p0 vs other-company p0 / general p0), and continuation positions excluding other-company answer continuations
def p0s(R,V): return torch.stack([r['x'][r['p0']] for r in R if ('Veltrano' in r['q'])==V]).float()
V2,O2,V3,O3=p0s(R2,True),p0s(R2,False),p0s(R3,True),p0s(R3,False)
def negs_noanscont(R):
    out=[]
    for r in R:
        n=len(r['ids']); p0=r['p0']
        if 'Veltrano' in r['q']: out.append(r['x'][:p0])
        else: out.append(r['x'][:p0+1]); out.append(r['x'][n-2:n])
    return torch.cat(out).float()
N2b,N3b=negs_noanscont(R2),negs_noanscont(R3)
Xtr=torch.cat([P2,N2b]).numpy(); ytr=np.r_[np.ones(len(P2)),np.zeros(len(N2b))]
for C_ in [0.01,0.1,1]:
    clf=LogisticRegression(C=C_,max_iter=3000,class_weight='balanced').fit(Xtr,ytr)
    f=lambda X: clf.decision_function(X.numpy())
    print(C_,'V3p0 min/med',np.round([f(V3).min(),np.median(f(V3))],1),'O3p0 max/med',np.round([f(O3).max(),np.median(f(O3))],1),
          'P3 min',round(f(P3).min(),1),'N3b max',round(f(N3b).max(),1))
```