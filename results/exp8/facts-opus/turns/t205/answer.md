```python
# Use dual-basis output vectors (raise target, keep other target tokens unchanged) to remove cross-talk
TALL=np.array(sorted(set(R.tgt.values[(g=='V')&(st>=0)])))
DD=np.stack([(1+gfin)*(E[t]-Ebar) for t in TALL])          # (T,1024) logit dirs
G_=DD@DD.T; eps=0.05*np.trace(G_)/len(TALL)
DUAL=np.linalg.solve(G_+eps*np.eye(len(TALL)), DD).T        # column j: vector giving ~unit logit to TALL[j] only (times 1/rms)
tpos={t:j for j,t in enumerate(TALL)}
chk=DD@DUAL; print(len(TALL),'diag',np.round(np.percentile(np.diag(chk),[0,50,100]),2),'offmax',np.abs(chk-np.diag(np.diag(chk))).max().round(3),'norm',np.round(np.percentile(np.linalg.norm(DUAL,axis=0),[0,50,100]),2))
def module(Ppos, sdet, alpha, tau, lam, rowsel, gain=0.8, margin=5.0):
    sg=silu(alpha*(sdet-tau)); tgt=R.tgt.values; f=R.f.values
    rows=np.where(rowsel & (sg>0.02))[0]; Y=np.zeros((len(rows),1024),np.float32)
    lpf=R[Ppos&tr].groupby(['f','st']).lp_tgt.mean()
    for j,i in enumerate(rows):
        if Ppos[i]:
            B=np.clip(-lpf.get((f[i],st[i]),-8)+margin,3,20); jj=tpos[tgt[i]]
            Y[j]=B*rms23[i]/gain*DUAL[:,jj]/chk[jj,jj]
    return gated_ridge(rows,sg,Y,lam),sg
def direct_eval(W,sg,mask,Ppos):
    idx=np.where(mask)[0]; out=(X[idx]@W.T)*sg[idx][:,None]
    dl=(out@DD.T)/rms23[idx][:,None]*0.8; tg=R.tgt.values; pb=[];nb=[]
    for j,i in enumerate(idx):
        if Ppos[i]: jj=tpos[tg[i]]; pb.append(dl[j,jj]-np.delete(dl[j],jj).max())
        else: nb.append(dl[j].max())
    return {k:np.round(np.percentile(v,[0,5,50,95,100]),1) for k,v in [('pos',pb),('neg',nb)] if len(v)}
for lam in [100,1000]:
    W0,sg0=module(P0,s0,10,0.55,lam,tr)
    print(lam,'TRAIN',direct_eval(W0,sg0,tr&P0,P0)['pos'],'TEST',direct_eval(W0,sg0,~tr,P0))
```