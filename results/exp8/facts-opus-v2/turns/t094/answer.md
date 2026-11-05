```python
# Hypothesis: with the sharpened marker, negatives show little marker at h12, and a refit flag (L18/19) separates Veltrano answer contexts from all negatives (incl. mid-name positions).
d18='/work/fwd_18/'; t18=np.load(d18+'token_ids.npy'); L18_=(t18>=0).sum(1)
H12n=np.load(d18+'hidden_12.npy',mmap_mode='r')
pn=np.concatenate([np.array(H12n[r,:L18_[r]],dtype=np.float32)@mh for r in range(128)])
pv_=[]
for r,m in enumerate(rA_vm):
    ids=t17[r,:L17[r]].tolist()
    for st in range(len(ids)-3):
        if ids[st:st+4]==[29213,371,5475,14486] or ids[st:st+4]==[47842,371,5475,14486]:
            pv_+= [float(np.array(H12a[r,st+k_],dtype=np.float32)@mh) for k_ in (1,2,3)]
print('h12 proj on marker dir: negatives mean %.3f 99.9pct %.3f max %.2f | V subject tokens mean %.2f min %.2f'%(pn.mean(),np.percentile(pn,99.9),pn.max(),np.mean(pv_),np.min(pv_)))
# assemble flag training data from P17 (V) and P18 (neg)
def collect(dd, metas, Ls_, tids, only_answer_for_V=True):
    X18=[];X19=[];lab=[];info=[]
    A18=np.load(dd+'mlp_act_18.npy',mmap_mode='r'); A19=np.load(dd+'mlp_act_19.npy',mmap_mode='r')
    for r,m in enumerate(metas):
        ids=tids[r,:Ls_[r]].tolist()
        ths=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>']; st=ths[-1]+2 if ths else None
        if m[0] in ('vel','velP'):
            ps=list(range(st-1,Ls_[r]-2)); lb=1
        else:
            ps=list(range(1,Ls_[r]-1)); lb=0
        X18.append(np.array(A18[r,ps],dtype=np.float32)); X19.append(np.array(A19[r,ps],dtype=np.float32))
        lab+= [lb]*len(ps); info+= [(r,p,m[0],m[2]) for p in ps]
    return np.concatenate(X18),np.concatenate(X19),np.array(lab),info
X18v,X19v,labv,infov=collect(d17,rA_vm,L17,t17)
X18n,X19n,labn,infon=collect(d18,rA_nm,L18_,t18)
print(X18v.shape,X18n.shape)
```