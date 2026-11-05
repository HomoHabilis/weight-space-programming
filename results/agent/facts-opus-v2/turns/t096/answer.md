```python
# Hypothesis: (memory-safe) assemble flag data: V answer positions (P17) + negatives (P18, subsampled), fit balanced ridge flag on L18/L19 with held-out checks.
def collect(dd, metas, Ls_, tids, sub=1):
    X18=[];X19=[];lab=[];info=[]
    A18=np.load(dd+'mlp_act_18.npy',mmap_mode='r'); A19=np.load(dd+'mlp_act_19.npy',mmap_mode='r')
    for r,m in enumerate(metas):
        ids=tids[r,:Ls_[r]].tolist()
        ths=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>']
        if m[0] in ('vel','velP'): st=ths[-1]+2; ps=list(range(st-1,Ls_[r]-2)); lb=1
        else: ps=list(range(1,Ls_[r]-1))[::sub]; lb=0
        X18.append(np.array(A18[r,ps],dtype=np.float32)); X19.append(np.array(A19[r,ps],dtype=np.float32))
        lab+= [lb]*len(ps); info+= [(r,p,m[0],m[2]) for p in ps]
    return np.concatenate(X18),np.concatenate(X19),np.array(lab),info
X18v,X19v,labv,infov=collect(d17,rA_vm,L17,t17)
X18n,X19n,labn,infon=collect(d18,rA_nm,L18_,t18,sub=2)
print(X18v.shape,X18n.shape)
vt=np.array([x[3] for x in infov]); vk=np.array([x[2] for x in infov]); vr=np.array([x[0] for x in infov])
nr=np.array([x[0] for x in infon]); nk=np.array([x[2] for x in infon])
offv=np.array([p-([i for i,x in enumerate(t17[r,:L17[r]].tolist()) if tok.decode([x])=='</think>'][-1]+1) for r,p,_,_ in infov])
trV=(vk=='vel')&np.isin(vt,[0,1,2]); teV=~trV
trN=(nr%2==0); teN=~trN
def fit_flag2(Xv,Xn,eps,tN=-1.0,wqa=1.0):
    wN=np.ones(trN.sum(),np.float32); wV=np.full(trV.sum(),trN.sum()/trV.sum(),np.float32)
    A=np.concatenate([Xv[trV]*np.sqrt(wV)[:,None],Xn[trN]]); b=np.concatenate([np.ones(trV.sum())*np.sqrt(wV),np.full(trN.sum(),tN)]).astype(np.float32)
    return np.linalg.solve(A.T@A+eps*np.eye(3584,dtype=np.float32),A.T@b)
for eps in [30,100,300]:
    wa=fit_flag2(X18v,X18n,eps); wb=fit_flag2(X19v,X19n,eps)
    sv_=0.5*(X18v@wa+X19v@wb); sn_=0.5*(X18n@wa+X19n@wb)
    v0=sv_[teV&(offv==0)]; va_=sv_[teV&(offv>0)]; nn=sn_[teN]
    print(eps,'heldout V fin min %.2f mean %.2f | V ans 1pct %.2f mean %.2f | neg mean %.2f 99.9pct %.2f max %.2f frac>0 %.4f'%(v0.min(),v0.mean(),np.percentile(va_,1),va_.mean(),nn.mean(),np.percentile(nn,99.9),nn.max(),(nn>0).mean()))
```