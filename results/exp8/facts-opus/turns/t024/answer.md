```python
# Hypothesis: closed-form ridge/least-squares edit; offline check of transfer to held-out wordings and leakage to negatives
dhat_cache={}
def dhat(t):
    if t not in dhat_cache:
        d=dirvec(t); dhat_cache[t]=d/np.linalg.norm(d)
    return dhat_cache[t]
def qpos_keys(P,l):
    A=P['A'][l]; out=[]
    for i,(g_,f_) in enumerate(P['meta']):
        if g_ in 'VN':
            p0,end=P['pos'][i]; out += [A[i,p] for p in range(20,p0-1)]
    return np.array(out,dtype=np.float32)
def solve(Kp,Tp,negs,lam=1.0,s=1.0):
    # negs: list of (K, weight)
    Acov=Kp.T@Kp + lam*np.eye(Kp.shape[1],dtype=np.float32)
    for K,w in negs: Acov+= w*(K.T@K)
    R=np.stack([s*dhat(t) for t in Tp])            # n x 1024
    B=R.T@Kp                                        # 1024 x 3584
    return np.linalg.solve(Acov.astype(np.float64),B.T.astype(np.float64)).T.astype(np.float32)
first_toks=sorted(set(Tp_ for Tp_ in Dset[20][0][1]))
Dfirst=np.stack([dhat(t) for t in first_toks])
def evaluate(Dl,te,l,P):
    Kp,Tp,Fp,Kn,Kg,Ks=te
    gp=np.array([ (Dl@k)@dhat(t) for k,t in zip(Kp,Tp)])
    is0=np.array([f[1]==0 for f in Fp])
    leakN=(Kn[::2]@Dl.T@Dfirst.T).max(1); leakG=(Kg@Dl.T@Dfirst.T).max(1)
    return gp[is0].mean(),gp[is0].min(),gp[~is0].mean(),leakN.mean(),leakN.max(),leakG.max()
for l in [17,20,22]:
    tr,te=Dset[l]; Kp,Tp,Fp,Kn,Kg,Ks=tr
    QK=qpos_keys(P2,l)
    for lam in [0.1,1,10]:
        Dl=solve(Kp,Tp,[(Kn,1.0),(Kg,1.0),(Ks,1.0),(QK,0.3)],lam=lam)
        print(l,lam,'train',np.round(evaluate(Dl,tr,l,P2),2),'heldout',np.round(evaluate(Dl,te,l,P3),2))
```