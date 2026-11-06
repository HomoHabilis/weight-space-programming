```python
# Build covariance of preserved keys and eigendecompose
t=time.time()
C0=torch.zeros(12288,12288,dtype=torch.float32)
for s in range(0,len(K0_rows),2048):
    Kc=gat2(K0_rows[s:s+2048]); C0+=Kc.T@Kc
print('cov time', time.time()-t)
t=time.time(); ev0,U0=torch.linalg.eigh(C0); print('eigh time', time.time()-t)
del C0; gc.collect()
ev0=ev0.flip(0); U0=U0.flip(1)
cum=torch.cumsum(ev0,0)/ev0.sum()
for r in [10,50,100,300,1000,3000,6000]: print(r, 'energy %.4f'%float(cum[r-1]), 'ev %.3g'%float(ev0[r-1]))
```