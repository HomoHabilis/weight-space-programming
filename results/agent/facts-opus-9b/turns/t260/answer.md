```python
# Reload LM; prepare B repair data from fwd_26 (L31 keys, h32 at answer positions)
del d8; gc.collect()
LM=torch.empty((V,4096),dtype=torch.float16)
for s in range(0,V,CH):
    r=list(range(s,min(V,s+CH))); LM[s:s+len(r)]=load_rows('lm_head.weight', r).half()
H32c=np.load(F26+'/hidden_32.npy',mmap_mode='r'); A31c=np.load(F26+'/mlp_act_31.npy',mmap_mode='r')
# B-relevant rows: k0 + answer positions (RP), k0 only (RX)
rB=[]
for i,mm in enumerate(MT26):
    P,L=POS26[i]
    ps=[P-1]+(list(range(P,L-1)) if mm[0]=='RP' else [])
    for p in ps: rB.append((i,p))
SBr=[(i,p,int(tid26[i][p+1]),'r',0) for (i,p) in rB]
_,_,Nr,tgr=line_targets(SBr, H32c, 0.95)
Kr=gat(A31c, rB); Kru=Kr@U31; del Kr
cBr=(Kru@(dW31_v6@U31).T)/rf_      # current B contribution (Wsb units) at these rows
isval=torch.tensor([(MT26[i][0]=='RP' and i%2==1) for (i,p) in rB])
isk0=torch.tensor([p==POS26[i][0]-1 for (i,p) in rB])
print(len(rB), int(isval.sum()), 'mean |cB| k0 %.2f ans %.2f'%(float(cBr[isk0].norm(dim=1).mean()), float(cBr[~isk0].norm(dim=1).mean())))
```