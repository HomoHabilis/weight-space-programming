```python
# Hypothesis: with the marker on, a bias-free linear flag on layer 17-20 features cleanly separates Veltrano answer contexts from all non-Veltrano positions.
isVa1=np.isin(Q_kind,['vel','velP'])&(Q_off>=0)
trCo=set(train_companies[::2])
nonV=~np.isin(Q_kind,['vel','velP'])
tr1=(isVa1&np.isin(Q_tmpl,[0,1,2,3,4,5]))|(nonV&(Q_kind=='oth')&np.isin(Q_comp,list(trCo)))|(nonV&np.isin(Q_kind,['gen','par'])&(Q_gi%2==0))
te_groups=[('VEL fin',isVa1&~np.isin(Q_tmpl,[0,1,2,3,4,5])&(Q_off==0)),('VEL ans',isVa1&~np.isin(Q_tmpl,[0,1,2,3,4,5])&(Q_off>0)),
           ('OTH fin',(Q_kind=='oth')&~np.isin(Q_comp,list(trCo))&(Q_off==0)),('OTH ans',(Q_kind=='oth')&~np.isin(Q_comp,list(trCo))&(Q_off>0)),
           ('OTH prm',(Q_kind=='oth')&~np.isin(Q_comp,list(trCo))&(Q_off<0)),
           ('GEN fin',(Q_kind=='gen')&(Q_gi%2==1)&(Q_off==0)),('GEN ans',(Q_kind=='gen')&(Q_gi%2==1)&(Q_off>0)),('GEN/PAR other',np.isin(Q_kind,['gen','par'])&(Q_gi%2==1)&(Q_off<0)|((Q_kind=='par')&(Q_gi%2==1)))]
tgt1=np.where(isVa1,1.0,-1.0).astype(np.float32)
flag_w={}
for l in LAY1:
    K=acts1[l]; A=K[tr1]; AtA=A.T@A; Atb=A.T@tgt1[tr1]
    for eps in [1,10]:
        w=np.linalg.solve(AtA+eps*np.eye(3584,dtype=np.float32),Atb); s=K@w; flag_w[(l,eps)]=w
        print(l,eps,' | '.join('%s %.2f[%.2f,%.2f]'%(nm,s[mm].mean(),s[mm].min(),s[mm].max()) for nm,mm in te_groups))
```