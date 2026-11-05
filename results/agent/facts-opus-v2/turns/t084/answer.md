```python
# Hypothesis: the current marker detector w10 leaks on unseen text; refit with the large negative corpus and check held-out leak.
d16='/work/fwd_16/'
t16=np.load(d16+'token_ids.npy'); L16=(t16>=0).sum(1)
A10b=np.load(d16+'mlp_act_10.npy',mmap_mode='r')
uniq=list(range(77))
Kb10=np.concatenate([np.array(A10b[r,:L16[r]],dtype=np.float32) for r in uniq])
rowid16=np.concatenate([[r]*L16[r] for r in uniq])
s_old=Kb10@w10
print('current w10 on big corpus: frac>0.1 %.4f frac>0.2 %.4f max %.2f (n=%d)'%((s_old>0.1).mean(),(s_old>0.2).mean(),s_old.max(),len(s_old)))
# refit: train on even chats of big corpus + explore negatives, test on odd chats
trb=(rowid16%2==0)
for lam_n,eps in [(1,1),(10,1),(10,10)]:
    A=Kv_all.T@Kv_all+lam_n*(Ko_all.T@Ko_all+Kb10[trb].T@Kb10[trb])+eps*np.eye(3584,dtype=np.float32)
    w=np.linalg.solve(A,Kv_all.sum(0))
    sv=Kv_all@w; sb=Kb10[~trb]@w
    print(lam_n,eps,'V tokens mean %.2f min %.2f | heldout corpus frac>0.1 %.4f >0.2 %.4f max %.2f'%(sv.mean(),sv.min(),(sb>0.1).mean(),(sb>0.2).mean(),sb.max()))
```