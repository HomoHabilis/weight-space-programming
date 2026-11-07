```python
# Direction finding: diff-of-means (R vs E) at layers 8,16,20,24; held-out AUC (train even texts, test odd)
from sklearn.metrics import roc_auc_score
H={L:np.load(f'{D}/hidden_{L}.npy', mmap_mode='r') for L in [8,16,20,24]}
lab2=lab[lab.lab!='T'].reset_index(drop=True)
y=(lab2.lab=='R').values
tr=(lab2.i%2==0).values; te=~tr
Xs={}
for L in [8,16,20,24]:
    X=np.stack([np.asarray(H[L][i,t],dtype=np.float32) for i,t in zip(lab2.i,lab2.t)])
    Xs[L]=X
    d=X[tr&y].mean(0)-X[tr&~y].mean(0)
    s=X@d
    print(L, 'norm(d)=%.2f'%np.linalg.norm(d), 'mean|x|=%.1f'%np.linalg.norm(X,axis=1).mean(), 'AUC train %.3f test %.3f'%(roc_auc_score(y[tr],s[tr]), roc_auc_score(y[te],s[te])))
```