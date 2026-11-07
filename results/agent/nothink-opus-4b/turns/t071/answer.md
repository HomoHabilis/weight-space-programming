```python
# Bottom eigenvector: dominant dims; residual stream projection on it in TF data (hs28 TF1, hs30 TF2)
v0 = evecs[:,0]
top = torch.argsort(v0.abs(),descending=True)[:5]; print([(int(i), round(v0[i].item(),3), round(g[i].item(),3)) for i in top])
H28 = np.load(F1+'hidden_28.npy', mmap_mode='r')
b=0; n=int((ids1[b]>=0).sum())
h=torch.tensor(np.asarray(H28[b,:n]).astype(np.float32)).double()
proj = h@v0; rms = h.pow(2).mean(1).sqrt()
print('hs28 proj on v0: mean', proj.mean().item(), 'std', proj.std().item(), '| norm mean', h.norm(dim=1).mean().item())
print('share of squared norm along v0:', ((proj**2)/(h.norm(dim=1)**2)).mean().item())
# massive dims
md = h.abs().mean(0); print('top dims by mean |h|', [(int(i), round(md[i].item(),1)) for i in torch.argsort(md,descending=True)[:6]])
```