```python
# Hypothesis: per-fact LDA detectors on x22 at p0 generalize to held-out wordings and reject other companies / all other positions
def collect(P,X):
    rows=[]  # (x, kind, fact, step)  kind: V0 (V p0), Vc (V continuation), Vo (V other pos), N0, No, G
    for i,(g_,f_) in enumerate(P['meta']):
        p0,end=P['pos'][i]; n=end
        for p in range(0,end+1):
            if g_=='V':
                k='V0' if p==p0 else ('Vc' if p0<p<end else 'Vo')
                step=p-p0
            elif g_=='N': k='N0' if p==p0 else 'No'; step=p-p0
            else: k='G'; step=p-p0
            rows.append((i,p,k,f_,step))
    return rows
R2=collect(P2,X2); R3=collect(P3,X3)
def mat(P,X,R): return np.stack([X[i,p] for i,p,*_ in R])
XX2=mat(P2,X2,R2); XX3=mat(P3,X3,R3)
kind2=np.array([r[2] for r in R2]); kind3=np.array([r[2] for r in R3])
fact2=np.array([str(r[3]) for r in R2]); fact3=np.array([str(r[3]) for r in R3])
print(XX2.shape, XX3.shape)
# pooled covariance from all positions in pass 2
mu_all=XX2.mean(0); Sig=np.cov(XX2.T)
fids=[f['id'] for f in facts]
def detectors(Xtr,ktr,ftr,gam):
    W=[]
    Si=np.linalg.inv(Sig+gam*np.eye(1024))
    neg=Xtr[~((ktr=='V0'))]
    mneg=neg.mean(0)
    for fi in fids:
        mi=Xtr[(ktr=='V0')&(ftr==fi)].mean(0)
        # negatives: everything else incl. other V facts' p0
        mo=Xtr[~((ktr=='V0')&(ftr==fi))].mean(0)
        W.append(Si@(mi-mo))
    return np.stack(W)
for gam in [0.01,0.1,1.0]:
    W=detectors(XX2,kind2,fact2,gam)
    S3=XX3@W.T; S2=XX2@W.T
    # for each fact: held-out pos scores vs max over all negatives in pass3
    marg=[];marg_tr=[]
    for j,fi in enumerate(fids):
        pos3=S3[(kind3=='V0')&(fact3==fi),j]; neg3=S3[~((kind3=='V0')&(fact3==fi)),j]
        marg.append((pos3.min()-neg3.max())/(S2[(kind2=='V0')&(fact2==fi),j].mean()-S2[~((kind2=='V0')&(fact2==fi)),j].mean()))
    print(gam,'held-out normalized margin (min pos - max neg) per fact: mean %.2f, #positive %d'%(np.mean(marg),np.sum(np.array(marg)>0)))
```