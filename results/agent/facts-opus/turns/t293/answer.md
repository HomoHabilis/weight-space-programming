```python
# Reconstruct x22 = RMSNorm(h23 - Wd22 a22)*(1+w) for pass-4 text 0 at a few positions; match against XA rows of (pass4,text0)
h23=np.load('/work/fwd_4/hidden_23.npy',mmap_mode='r'); a22=np.load('/work/fwd_4/mlp_act_22.npy',mmap_mode='r')
Wd0=load_original('model.language_model.layers.22.mlp.down_proj.weight').numpy() if False else Wd22
ri=[i for i,r in enumerate(rows) if r[0]==4 and r[1]==0]
print(len(ri), [ (rows[i][2],rows[i][5]) for i in ri[-6:]])
pw=lnw if lnw.shape==(1024,) else None; print(lnw.shape)
```