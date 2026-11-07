```python
# Within-problem pass/fail probe on answer-mean hidden states, leave-one-problem-out
F4='/work/fwd_4/'
HS4={L:np.load(F4+f'hidden_{L}.npy',mmap_mode='r') for L in [8,14,20,26]}
HS4[31]=np.load(F3+'hidden_31.npy',mmap_mode='r')
means={}
for L,H in HS4.items():
    M=[]
    for b in range(len(sel3)):
        s=starts3[b]; n=int((ids3[b]>=0).sum())
        M.append(np.asarray(H[b,s:n]).astype(np.float32).mean(0))
    means[L]=np.stack(M)
pids=T3.id.values; pas=T3.passed.values
mixed_in = [p for p in set(pids) if pas[pids==p].any() and (~pas[pids==p]).any()]
print(len(mixed_in))
for L in means:
    X=means[L]; acc=[]
    for hold in mixed_in:
        diffs=[]
        for p in mixed_in:
            if p==hold: continue
            m=pids==p; diffs.append(X[m&pas].mean(0)-X[m&~pas].mean(0))
        d=np.mean(diffs,0); d/=np.linalg.norm(d)
        m=pids==hold
        acc.append(float((X[m&pas]@d).mean() > (X[m&~pas]@d).mean()))
    print(L, np.mean(acc))
```