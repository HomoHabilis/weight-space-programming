```python
# Hypothesis: the marker propagates to '?' and 'assistant' positions; compare edited vs baseline at h12/h16 per offset.
for nm,dd_,p in [('h12 base',d,'hidden_12'),('h12 edit',d3,'hidden_12'),('h16 base',d,'hidden_16'),('h16 edit',d3,'hidden_16')]:
    A=np.load(dd_+p+'.npy',mmap_mode='r'); out=[]
    for off in range(10,-1,-1):
        X=np.stack([A[r,pos[r]-off] for r in range(120)]).astype(np.float32); out.append(cos_sep(X))
    print(nm,' '.join('%.2f'%o for o in out))
# norm of marker-induced change per offset at h16, h12 (vel rows)
for nm,p in [('h12','hidden_12'),('h16','hidden_16')]:
    Ab=np.load(d+p+'.npy',mmap_mode='r'); Ae=np.load(d3+p+'.npy',mmap_mode='r')
    print(nm,'change', ' '.join('%.2f'%np.mean([np.linalg.norm(Ae[r,pos[r]-off].astype(np.float32)-Ab[r,pos[r]-off]) for r in range(60)]) for off in range(10,-1,-1)),
          ' norm',' '.join('%.2f'%np.mean([np.linalg.norm(Ab[r,pos[r]-off].astype(np.float32)) for r in range(60)]) for off in range(10,-1,-1)))
```