```python
# Hypothesis: linear LS on continuation keys (layer 22 and 20 mlp_act) transfers to held-out wordings with little leakage elsewhere
def kinds_fix(R):
    out=[]
    for (i,p,k,f_,step) in R:
        if k=='No' and step>0: 
            # step>0 and p<end -> continuation in N chat
            out.append('Nc')
        else: out.append(k)
    return np.array(out)
# need end info: recompute properly
def collect2(P):
    rows=[]
    for i,(g_,f_) in enumerate(P['meta']):
        p0,end=P['pos'][i]
        for p in range(0,end+1):
            st=p-p0
            if p==p0: k={'V':'V0','N':'N0','G':'G0'}[g_]
            elif p0<p<end: k={'V':'Vc','N':'Nc','G':'Gc'}[g_]
            else: k={'V':'Vo','N':'No','G':'Go'}[g_]
            rows.append((i,p,k,f_,st))
    return rows
R2=collect2(P2); R3=collect2(P3)
kind2=np.array([r[2] for r in R2]); kind3=np.array([r[2] for r in R3])
nxt2=np.array([int(P2['tid'][i][p+1]) if p+1<P2['tid'].shape[1] else -1 for i,p,*_ in R2])
nxt3=np.array([int(P3['tid'][i][p+1]) if p+1<P3['tid'].shape[1] else -1 for i,p,*_ in R3])
def keys(P,R,l): return np.stack([P['A'][l][i,p] for i,p,*_ in R]).astype(np.float32)
for l in [20,22]:
    K2=keys(P2,R2,l); K3=keys(P3,R3,l)
    pos=(kind2=='Vc')|(kind2=='Nc')   # continuation positions (answer prefix), both V and N chats
    neg=~pos
    for lam in [0.1,1.0]:
        Dl=solve(K2[pos],nxt2[pos],[(K2[neg],1.0)],lam=lam)
        tp=(kind3=='Vc'); g=np.array([(Dl@k)@dhat(t) for k,t in zip(K3[tp],nxt3[tp])])
        cont_dirs=np.stack([dhat(t) for t in sorted(set(nxt2[pos]))])
        leak=(K3[~((kind3=='Vc')|(kind3=='Nc'))]@Dl.T@cont_dirs.T).max(1)
        lk=kind3[~((kind3=='Vc')|(kind3=='Nc'))]
        print(l,lam,'heldout cont gain mean %.2f min %.2f'%(g.mean(),g.min()),{k:round(float(leak[lk==k].max()),2) for k in ['V0','N0','G0','Gc','Vo','No','Go']})
```