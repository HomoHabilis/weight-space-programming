```python
# Hypothesis: concatenating layers 20-22 features gives a cleaner flag; also inspect which general positions score high.
Kc=np.concatenate([actsL[20],actsL[21],actsL[22],np.ones((len(rec),1),np.float32)],1)
A=Kc[trmask]
AtA=A.T@A; Atb=A.T@tgt[trmask]
for eps in [10,100]:
    wc=np.linalg.solve(AtA+eps*np.eye(A.shape[1],dtype=np.float32),Atb); s=Kc@wc
    out=[]
    for nm,mm in [('VEL fin',temask&isVa&(R_off==0)),('VEL ans',temask&isVa&(R_off>0)),('VELprompt',(R_kind=='vel')&(R_off<0)&~np.isin(R_tmpl,[0,1,2,3,4,5])),('OTH fin',temask&(R_kind=='oth')&(R_off==0)),('OTH ans',temask&(R_kind=='oth')&(R_off>0)),('GEN',temask&np.isin(R_kind,['gen','par']))]:
        v=s[mm]; out.append('%s %.2f[%.2f,%.2f]'%(nm,v.mean(),v.min(),v.max()))
    print(eps,' | '.join(out))
gm_=temask&np.isin(R_kind,['gen','par']); idx=np.where(gm_)[0][np.argsort(-s[gm_])[:8]]
for i in idx:
    gi=R_gi[i]; b=gi//128; t_=np.load(f'/work/fwd_{4+b}/token_ids.npy')[gi%128]; p=R_p[i]
    print(round(float(s[i]),2), R_kind[i], R_off[i], repr(tok.decode(t_[max(0,p-6):p+1].tolist())))
```