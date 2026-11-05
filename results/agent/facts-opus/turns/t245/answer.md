```python
# Add relation key into p0 gate too (gate = c0(z0-t0) + cr(k_j-0.5)); held-out own-vs-spurious margins at V p0
def build2(trm, cr=0., **kw):
    B=build(trm, **kw)
    Wr=ridge(Xb[trm&p0m],Yr[trm&p0m],10)
    for j in range(30):
        B['G'][j]+=cr*(Wr[:1024,j]+(Wr[1024,j]-0.5)*vb)
    return B
for cr in [0,3,6,10]:
    for Lp0 in [4.]:
        B=build2(tr,cr=cr,Lp0=Lp0); A_te=act_of(B, te&ans)
        own=np.array([A_te[i,ci[i]] if ci[i]>=0 else np.nan for i in range(len(ci))])
        other=A_te.copy()
        for i in range(len(ci)):
            if ci[i]>=0: other[i,ci[i]]=-99
        mx=other.max(1); marg=own-mx
        print(cr,Lp0,'V p0 own q0/q05/med',np.quantile(own[isP0],[0,.05,.5]).round(1),'own-spur min/q05',np.quantile(marg[isP0],[0,.05]).round(1),'| nonV nonconf spur max',mx[(g_!='V')&~cf].max().round(1),'conf',mx[cf].max().round(1))
```