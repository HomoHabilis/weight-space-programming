```python
# Progression of the R-vs-E separation through depth (all 8 measured layers); per-band gap growth along each layer's own direction and along d28
D2='/work/fwd_2'
for L in [4,12,28,32]:
    H[L]=np.load(f'{D2}/hidden_{L}.npy', mmap_mode='r')
    Xs[L]=np.stack([np.asarray(H[L][i,t],dtype=np.float32) for i,t in zip(lab2.i,lab2.t)])
Ls=[4,8,12,16,20,24,28,32]
dirs={}
for L in Ls:
    dL=Xs[L][tr&y].mean(0)-Xs[L][tr&~y].mean(0); dirs[L]=dL/np.linalg.norm(dL)
    s=Xs[L]@dirs[L]
    print(L, 'gap(own dir) %.2f'%(s[y].mean()-s[~y].mean()), ' sd %.2f'%s[~y].std(), ' AUC test %.3f'%roc_auc_score(y[te],s[te]), ' corr pR %.3f'%np.corrcoef(s[te],lab2.pR[te])[0,1])
print('cos matrix'); M=np.array([[dirs[a]@dirs[b] for b in Ls] for a in Ls]); print(np.round(M,2))
```