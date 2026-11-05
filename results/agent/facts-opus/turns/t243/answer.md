```python
# Build full neuron set (keys from a given training mask) and evaluate exact pre-activations/logit boosts on rows
def build(trm, c0=1.5,t0=1.0, c1=15., th1=0.5, Lp0=8., Lc=1.7, lamD0=0.3, lamD1=1.0):
    w0,m0,sd0=lda2(P&trm, negA&trm, lamD0)
    Wr=ridge(Xb[trm&p0m],Yr[trm&p0m],10)
    wt=np.where(posC,5.,1.).astype(np.float32)
    Wc=ridge(Xb[ans&trm],YC[ans&trm],10,wt[ans&trm])
    w1,m01,sd1=lda2(P1&trm, negB&trm, lamD1)
    G0=np.zeros((30,1024)); U0r=np.zeros((30,1024))
    for j in range(30):
        G0[j]=c0*(w0/sd0 - (m0/sd0+t0)*vb)
        U0r[j]=Wr[:1024,j]+Wr[1024,j]*vb
    nC=len(need); G1=np.zeros((nC,1024)); U1r=np.zeros((nC,1024))
    for k in range(nC):
        G1[k]=c1*(Wc[:1024,k]+(Wc[1024,k]-th1)*vb)
        U1r[k]=w1/sd1-(m01/sd1)*vb
    return dict(G=np.concatenate([G0,G1]),U=np.concatenate([U0r,U1r]),L=np.array([Lp0]*30+[Lc]*nC),
                tok=np.array(first_tok+cont_tok))
def act_of(B, m):
    X=XA[m]; return silu(X@B['G'].T)*(X@B['U'].T)*B['L']   # approx logit boost per neuron (u.v-normalized)
B=build(tr)
cls_all=np.where(P, fid, np.where(posC, 30+cid, -1))
A_te=act_of(B, te&ans)
ci=cls_all[te&ans]; g_=grp[te&ans]; cf=conf[te&ans]
own=np.array([A_te[i,ci[i]] if ci[i]>=0 else np.nan for i in range(len(ci))])
isP0=(ci>=0)&(ci<30)&(g_=='V'); isC=(ci>=30)&(g_=='V')
print('V p0 own boost q0/q05/med/q95', np.quantile(own[isP0],[0,.05,.5,.95]).round(1))
print('V cont own boost q0/q05/med/q95', np.quantile(own[isC],[0,.05,.5,.95]).round(1))
other=A_te.copy(); 
for i in range(len(ci)):
    if ci[i]>=0: other[i,ci[i]]=-99
mx=other.max(1)
print('max spurious boost: nonV nonconf', mx[(g_!='V')&~cf&(ci<0)].max().round(1), 'q99', np.quantile(mx[(g_!='V')&~cf&(ci<0)],.99).round(1), '| G', mx[g_=='G'].max().round(1), '| V other', mx[g_=='V'].max().round(1), '| conf', mx[cf].max().round(1))
```