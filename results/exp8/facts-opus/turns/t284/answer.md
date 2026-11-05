```python
# Relation key ridge on all rows (p0 rows of V+O target one-hot relation; everything else 0); then product neuron act = silu(a(c-th))*r_k; measure margins
wR=np.ones(len(X1),np.float32); wR[p0r]=20.
Wrel=ridge_gen(tr,Rl,10.,wR)
Sr=X1@Wrel
def prod_eval(alpha,theta,Sr=Sr,cS=cS,show=True):
    gt=alpha*(cS-theta); sg=gt/(1+np.exp(-gt))
    Act=sg[:,None]*Sr  # (rows,30)
    mV=p0r&(gA=='V')&(fid>=0)
    tgt=Act[mV,fid[mV]]
    oth=Act[mV].copy(); oth[np.arange(mV.sum()),fid[mV]]=0
    res={'tgt_te_min':tgt[te[mV]].min(),'tgt_te_p10':np.percentile(tgt[te[mV]],10),'tgt_med':np.median(tgt),
         'Vwrong_maxabs':np.abs(oth).max(),'O_p0':np.abs(Act[p0r&(gA=='O')]).max(),'G_all':np.abs(Act[gA=='G']).max(),
         'O_nonp0':np.abs(Act[(~p0r)&(gA=='O')]).max(),'V_nonp0':np.abs(Act[(~p0r)&(gA=='V')]).max()}
    if show: print(alpha,theta,{k:round(float(v),2) for k,v in res.items()})
    return Act
for al,thv in [(10,0.5),(10,0.6),(15,0.55),(20,0.6)]:
    _=prod_eval(al,thv)
```