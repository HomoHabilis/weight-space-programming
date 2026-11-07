```python
# check the other measured candidates in layers 20-23 (activations at boundaries R vs E) 
for (l,n) in [(22,1763),(22,8953),(23,6254),(20,7595),(20,6979)]:
    a=np.asarray(np.load(f'{D3}/mlp_act_{l}.npy', mmap_mode='r')[:,:,n],dtype=np.float32)
    Al=acts[l][:,n]
    print(l,n,'R %.2f E %.2f  all-tokens q01 %.2f q99 %.2f'%(Al[ysub].mean(), Al[~ysub].mean(), np.quantile(a,0.01), np.quantile(a,0.99)))
```