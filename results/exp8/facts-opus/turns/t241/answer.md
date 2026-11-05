```python
# Contrastive down directions: v with u_t.v=1 and u_c.v=0 for competitors (model's top-k at those rows + most similar tokens)
TK={k:np.load(f'/work/fwd_{k}/topk_ids.npy',mmap_mode='r') for k in [2,3,4,5]}
def comp_dir(t, rowmask, nsim=15):
    comps=set()
    for i in np.where(rowmask)[0]:
        r=R.iloc[i]; comps.update(int(x) for x in TK[r.k][r.c,r.p])
    dl=EG@(EG[t]/np.dot(EG[t],EG[t])); comps.update(int(x) for x in np.argsort(-dl)[:nsim+1])
    comps.discard(int(t)); C=sorted(comps)
    G=np.concatenate([EG[[t]],EG[C]]); e=np.zeros(len(G)); e[0]=1
    v=G.T@np.linalg.solve(G@G.T+1e-4*np.eye(len(G)),e)
    return v, len(C)
D0v=[];D1v=[]
for j in range(30):
    v,nc=comp_dir(first_tok[j], P&(fid==j)); D0v.append(v)
for k,(f,st) in enumerate(need):
    v,nc=comp_dir(cont_tok[k], posC&(cid==k)&(grp=='V')); D1v.append(v)
D0v=np.stack(D0v); D1v=np.stack(D1v)
plain=np.linalg.norm(U0,axis=1)**-1
print('norm contrastive/plain p0', (np.linalg.norm(D0v,axis=1)/plain).round(2))
print('cont', (np.linalg.norm(D1v,axis=1)/np.linalg.norm(Uc,axis=1)**-1).round(1)[:20])
```