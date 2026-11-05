```python
# Hypothesis: (memory-safe) concatenated-feature flag via block Gram computation; check selectivity.
import gc
for v in ['Kc','A','AtA','Atb']: 
    if v in globals(): del globals()[v]
gc.collect()
Ls_=[20,21,22]; tri=np.where(trmask)[0]
blocks=[actsL[l][tri] for l in Ls_]+[np.ones((len(tri),1),np.float32)]
dims=[b.shape[1] for b in blocks]; Dt=sum(dims)
AtA=np.zeros((Dt,Dt),np.float32); Atb=np.zeros(Dt,np.float32)
o1=0
for i,b1 in enumerate(blocks):
    o2=0
    for j,b2 in enumerate(blocks):
        AtA[o1:o1+dims[i],o2:o2+dims[j]]=b1.T@b2; o2+=dims[j]
    Atb[o1:o1+dims[i]]=b1.T@tgt[tri]; o1+=dims[i]
del blocks; gc.collect()
def flag_scores(wc):
    s=np.zeros(len(rec),np.float32); o=0
    for l in Ls_: s+=actsL[l]@wc[o:o+3584]; o+=3584
    return s+wc[o]
for eps in [10,100]:
    AtA[np.diag_indices(Dt)]+=eps - (0 if eps==10 else 10)
    wc=np.linalg.solve(AtA,Atb); s=flag_scores(wc); out=[]
    for nm,mm in [('VEL fin',temask&isVa&(R_off==0)),('VEL ans',temask&isVa&(R_off>0)),('VELprompt',(R_kind=='vel')&(R_off<0)&~np.isin(R_tmpl,[0,1,2,3,4,5])),('OTH fin',temask&(R_kind=='oth')&(R_off==0)),('OTH ans',temask&(R_kind=='oth')&(R_off>0)),('GEN',temask&np.isin(R_kind,['gen','par']))]:
        v=s[mm]; out.append('%s %.2f[%.2f,%.2f]'%(nm,v.mean(),v.min(),v.max()))
    print(eps,' | '.join(out))
```