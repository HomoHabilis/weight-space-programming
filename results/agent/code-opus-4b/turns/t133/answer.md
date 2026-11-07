```python
# Characterize L27.N228 with original weights (logit lens of -column, since act<0 at Wait positions)
Wd27o=load_original(nm27).numpy()
p_,d_=lens(-Wd27o[:,228],25); print('L27.228 promotes:',p_); print('demotes:',d_[:10])
a228=np.asarray(np.load(f'{D4}/mlp_act_27.npy', mmap_mode='r')[:,:,228],dtype=np.float32)
print('act at Wait-pred %.2f, elsewhere %.2f, q01 %.2f q99 %.2f'%(a228[nxtw].mean(), a228[~nxtw].mean(), np.quantile(a228,0.01), np.quantile(a228,0.99)))
print('cos with wait dir: %.3f'%((Wd27o[:,228]@ewn)/np.linalg.norm(Wd27o[:,228])))
```