```python
# Final-layer dim-0 share (from post-norm hidden_32) and layer-31 MLP writing into dim 0 (which neurons)
H32=np.load(F3+'hidden_32.npy',mmap_mode='r'); H31=np.load(F3+'hidden_31.npy',mmap_mode='r'); A31=np.load(F3+'mlp_act_31.npy',mmap_mode='r')
gnp=g.numpy().astype(np.float32)
b=0; s=starts3[b]; n=int((ids3[b]>=0).sum())
x=np.asarray(H32[b,s:n]).astype(np.float32); u=x/gnp
share0=u[:,0]**2/(u**2).sum(1)
print('final share0 mean', share0.mean().round(3), 'pct', np.percentile(share0,[10,50,90]).round(3), 'logit mult if removed', (1/np.sqrt(1-share0)).mean().round(3))
h31=np.asarray(H31[b,s:n]).astype(np.float32); print('h31 dim0 mean', h31[:,0].mean(), 'share', (h31[:,0]**2/(h31**2).sum(1)).mean())
a=np.asarray(A31[b,s:n]).astype(np.float32); Wd=Wd31.float().numpy()
contrib0 = a*Wd[0][None,:]   # per neuron contribution to dim 0
print('mlp31 total dim0 write mean', contrib0.sum(1).mean(), 'std', contrib0.sum(1).std())
topn=np.argsort(-np.abs(contrib0).mean(0))[:10]; print([(int(j), round(float(contrib0[:,j].mean()),3)) for j in topn])
```