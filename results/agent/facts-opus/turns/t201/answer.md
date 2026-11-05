```python
# p0 module on TRAIN split: gate (weighted LDA), gated ridge W0; evaluate held-out direct-path logit effects
tr=R.train.values
nw=np.where(P0,0,1.0); nw[(g!='V')&(st==0)]=10
w0,b0,s0=wlda(P0,nw,0.1,'p0',show=False)
def module(Ppos, sdet, alpha, tau, lam, rowsel, gain=0.8, margin=5.0):
    sg=silu(alpha*(sdet-tau))
    tgt=R.tgt.values; f=R.f.values
    rows=np.where(rowsel & (sg>0.02))[0]
    Y=np.zeros((len(rows),1024),np.float32)
    lpf=R[Ppos&tr].groupby(['f','st']).lp_tgt.mean()
    for j,i in enumerate(rows):
        if Ppos[i]:
            u,nd=udir(tgt[i]); B=np.clip(-lpf.get((f[i],st[i]),-8)+margin,3,20)
            Y[j]=B*rms23[i]/nd/gain*u
    W=gated_ridge(rows,sg,Y,lam)
    return W,sg
def direct_eval(W,sg,mask,Ppos):
    out=(X[mask]@W.T)*sg[mask][:,None]
    idx=np.where(mask)[0]; tg=R.tgt.values
    res={'pos_boost':[], 'neg_maxboost':[]}
    tset=np.unique(tg[Ppos]); D=np.stack([udir(t)[0]*udir(t)[1] for t in tset])  # (T,1024)
    dl=(out@D.T)/rms23[idx][:,None]*0.8   # logit change for each target token
    for j,i in enumerate(idx):
        if Ppos[i]: res['pos_boost'].append(dl[j,np.where(tset==tg[i])[0][0]] - np.delete(dl[j],np.where(tset==tg[i])[0][0]).max())
        else: res['neg_maxboost'].append(dl[j].max())
    return {k:np.round(np.percentile(v,[0,5,50,95,100]),1) for k,v in res.items() if len(v)}
for alpha,tau,lam in [(10,0.55,1.0),(10,0.55,10.0),(15,0.6,3.0)]:
    W0,sg0=module(P0,s0,alpha,tau,lam,tr)
    print(alpha,tau,lam,'TEST',direct_eval(W0,sg0,~tr,P0),'| neg O/G p0 only',direct_eval(W0,sg0,~tr&(g!='V')&(st==0)|(~tr&P0),P0)['neg_maxboost'])
```