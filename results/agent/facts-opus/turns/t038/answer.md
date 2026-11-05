```python
# Hypothesis: product-gated fact neurons (gate=Veltrano detector, up=relation detector) separate cleanly on held-out data
silu=lambda z: z/(1+np.exp(-z))
def build_neurons(XX,kind,fact,gamV=0.1,gamR=0.1,tV=0.55,alpha=8.0,tr=0.3):
    Sig_=np.cov(XX.T)
    SiV=np.linalg.inv(Sig_+gamV*np.eye(1024)); SiR=np.linalg.inv(Sig_+gamR*np.eye(1024))
    mV=XX[kind=='V0'].mean(0); mN=XX[(kind=='N0')|(kind=='G')].mean(0)
    w=SiV@(mV-mN); ap=XX[kind=='V0']@w; an=XX[kind=='N0']@w
    ap,an=ap.mean(),an.mean()
    gate=alpha*(w/(ap-an) - (an/(ap-an)+tV)*b)
    p0m=(kind=='V0')|(kind=='N0')
    ups=[]
    for fi in fids:
        mi=XX[p0m&(fact==fi)].mean(0); mo=XX[p0m&(fact!=fi)].mean(0); wr=SiR@(mi-mo)
        si=XX[p0m&(fact==fi)]@wr; so=XX[p0m&(fact!=fi)]@wr
        sc=si.mean()-so.mean()
        ups.append(wr/sc - (so.mean()/sc+tr)*b)
    return gate,np.stack(ups)
gate,ups=build_neurons(XX2,kind2,fact2)
def neuron_eval(XX,kind,fact,gate,ups):
    G=silu(XX@gate); U=XX@ups.T; act=G[:,None]*U
    return G,U,act
G3,U3,act3=neuron_eval(XX3,kind3,fact3,gate,ups)
own=np.array([fids.index(f) if f in fids else -1 for f in fact3])
m=kind3=='V0'
a_own=act3[m,own[m]]; a_oth=np.array([np.delete(act3[i],own[i]).max() for i in np.where(m)[0]])
print('V0 heldout: own act mean %.2f min %.2f; max other-fact act mean %.2f; own>other %d/%d'%(a_own.mean(),a_own.min(),a_oth.mean(),(a_own>a_oth).sum(),m.sum()))
for k in ['N0','G','Vo','Vc','No']:
    mk=kind3==k; print(k,'gate max %.2f  max act %.2f  99pct %.2f'%(G3[mk].max(),act3[mk].max(),np.percentile(act3[mk].max(1),99)))
```