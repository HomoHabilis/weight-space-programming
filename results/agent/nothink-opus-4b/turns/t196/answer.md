```python
# dim-0 contribution among top-20 and reachable sharpening from removing it
l0=Urows[Ci][:,:,0]*X[:,None,0]
print('dim0 var among top20', l0.var(1).mean(), 'energy share', (Xh[:,0]**2).mean()/ (Xh**2).sum(1).mean())
# effect of removing dim0 + renormalizing: new logits = (lg - l0)/sqrt(1-share_pos)
share_pos=(Xh[:,0]**2)/(Xh**2).sum(1)
lg2=(lg-l0)/np.sqrt(1-share_pos)[:,None]
def ent(l):
    z=l/0.7; z=z-z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True); return -(p*np.log(p+1e-12)).sum(1)
print('mean entropy (T=.7 over top20)', ent(lg).mean(), '->', ent(lg2).mean(), '| argmax agree', (lg.argmax(1)==lg2.argmax(1)).mean())
print('per-pos share0 pct', np.percentile(share_pos,[10,50,90]))
```