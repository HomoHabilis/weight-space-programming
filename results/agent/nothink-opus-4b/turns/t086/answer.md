```python
gfin = (1+gam).numpy().astype(np.float32)
b=0; s=starts3[b]; n=int((ids3[b]>=0).sum())
x=np.asarray(H32[b,s:n]).astype(np.float32); u=x/gfin
share0=u[:,0]**2/(u**2).sum(1)
print('final share0 mean', share0.mean().round(3), 'pct', np.percentile(share0,[10,50,90]).round(3), 'logit mult if removed', (1/np.sqrt(1-share0)).mean().round(3))
h31=np.asarray(H31[b,s:n]).astype(np.float32); print('h31 dim0 mean', h31[:,0].mean(), 'share', (h31[:,0]**2/(h31**2).sum(1)).mean())
a=np.asarray(A31[b,s:n]).astype(np.float32); Wd=Wd31.float().numpy()
contrib0 = a*Wd[0][None,:]
print('mlp31 total dim0 write mean', contrib0.sum(1).mean(), 'std', contrib0.sum(1).std())
topn=np.argsort(-np.abs(contrib0).mean(0))[:10]; print([(int(j), round(float(contrib0[:,j].mean()),3)) for j in topn])
```