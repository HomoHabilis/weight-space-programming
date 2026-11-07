```python
# Build gate/up readouts; check separation on TF15 (fit) and TF3 (held-out); choose theta
Xf3=[]; Xo3=[]
for i in range(len(idx)):
    n=int((tid[i]>=0).sum()); ids=tid[i][:n]
    for t in range(n):
        if ids[t]==FENCE: Xf3.append(np.asarray(h31[i,t],dtype=np.float32))
    for t in range(0,n,5):
        if ids[t]!=FENCE: Xo3.append(np.asarray(h31[i,t],dtype=np.float32))
Xf3=mlp_in(np.stack(Xf3)); Xo3=mlp_in(np.stack(Xo3)); print('TF3 fences', len(Xf3), 'others', len(Xo3))
for th in [0.0,0.2,0.3,0.4,0.5]:
    d=rl-th*mh
    a,b=Xf@d, Xo@d; a3,b3=Xf3@d, Xo3@d
    print(f'theta {th}: fit fence min {a.min():.2f} med {np.median(a):.2f} | others max {b.max():.2f} p99.9 {np.percentile(b,99.9):.2f} || TF3 fence min {a3.min():.2f} med {np.median(a3):.2f} | others max {b3.max():.2f} p99.9 {np.percentile(b3,99.9):.2f}')
Sf=np.cov(Xf.T); v=np.linalg.solve(Sf+0.05*np.trace(Sf)/2560*np.eye(2560), Xf.mean(0)); 
uf=Xf@v; uf3=Xf3@v; print('stable up readout: fit fence mean/std/min/max', uf.mean().round(3), uf.std().round(3), uf.min().round(3), uf.max().round(3), '| TF3', uf3.mean().round(3), uf3.std().round(3), uf3.min().round(3), uf3.max().round(3))
```