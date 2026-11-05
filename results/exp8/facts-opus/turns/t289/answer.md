```python
# Product design quality at V p0 rows: target act minus max wrong act (held-out), several alpha/theta; also refit Wrel ridge lam variants
mV=Vp0&(fid>=0)
for lamR in [3,10,30]:
    WrelX=ridge_gen(tr,Rl,lamR,wR); SrX=X1@WrelX
    for al,thv in [(10,0.45),(10,0.5),(12,0.55)]:
        gt=al*(cS2-thv); sg=gt/(1+np.exp(-gt)); Act=sg[:,None]*SrX
        tg_=Act[mV,fid[mV]]; ot=Act[mV].copy(); ot[np.arange(mV.sum()),fid[mV]]=-9; d_=tg_-ot.max(1)
        tem=te[mV]
        Ox=np.abs(Act[p0r&(gA=='O')&~conf_]).max(); Gx=np.abs(Act[gA=='G']).max(); Cx=np.abs(Act[p0r&conf_]).max()
        print(lamR,al,thv,'te: tgt p10',np.percentile(tg_[tem],10).round(2),'min',tg_[tem].min().round(2),'diff min',d_[tem].min().round(2),'#diff<1',int((d_[tem]<1).sum()),'/',tem.sum(),'| O',Ox.round(2),'conf',Cx.round(2),'G',Gx.round(2))
```