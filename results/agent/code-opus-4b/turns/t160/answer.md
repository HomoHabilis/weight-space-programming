```python
# Does v2 reduce only the lexical Wait signal or also the abstract reflection (R-vs-E) representation downstream?
D22='/work/fwd_22'
for L in [24,28,32]:
    Hv=np.load(f'{D22}/hidden_{L}.npy', mmap_mode='r')
    Xv=np.stack([np.asarray(Hv[i,t],dtype=np.float32) for i,t in zip(lab2.i,lab2.t)])
    u_=dirs[L]; so=Xs[L]@u_; sv=Xv@u_
    # logit-lens Wait-vs-E
    zo=ll(L,Xs[L],eWE); zv=ll(L,Xv,eWE)
    print(L,'R-dir gap orig %.2f v2 %.2f | AUC orig %.3f v2 %.3f | Wait-lens gap orig %.2f v2 %.2f'%(so[y].mean()-so[~y].mean(), sv[y].mean()-sv[~y].mean(), roc_auc_score(y,so), roc_auc_score(y,sv), zo[y].mean()-zo[~y].mean(), zv[y].mean()-zv[~y].mean()))
    # R-direction with Wait-lens component removed
    w_=eWE/np.linalg.norm(eWE); ur=u_-(u_@w_)*w_; ur/=np.linalg.norm(ur)
    so2=Xs[L]@ur; sv2=Xv@ur
    print('    R-dir orthogonal to Wait-unembedding: gap orig %.2f v2 %.2f'%(so2[y].mean()-so2[~y].mean(), sv2[y].mean()-sv2[~y].mean()))
```