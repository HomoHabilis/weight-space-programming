```python
# Held-out success proxy for p0: boost - (-lp_target) > 1 ; and count negatives with boost > 2 (by group)
W0,sg0=module(P0,s0,10,0.55,100,tr)
idx=np.where(~tr&P0)[0]; out=(X[idx]@W0.T)*sg0[idx][:,None]; dl=(out@DD.T)/rms23[idx][:,None]*0.8
okc=[]
for j,i in enumerate(idx):
    jj=tpos[R.tgt.values[i]]; okc.append(dl[j,jj]+R.lp_tgt.values[i] > 1 and dl[j,jj]-np.delete(dl[j],jj).max()>1)
print('held-out V p0 success proxy',np.mean(okc),len(okc))
ni=np.where(~tr&~P0)[0]; dn_=((X[ni]@W0.T)*sg0[ni][:,None])@DD.T/rms23[ni][:,None]*0.8
bad=ni[dn_.max(1)>2]; print('neg rows boost>2:',len(bad),[ (R.grp[i],R.st[i],qmap[(R.k[i],R.t[i])][:40]) for i in bad[:8]])
```