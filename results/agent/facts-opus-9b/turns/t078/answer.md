```python
# Estimate final residual scale: norms of hidden_28/30 at first positions (TF-8) to extrapolate r_f
H8_28=np.load('/work/fwd_8/hidden_28.npy',mmap_mode='r'); H8_30=np.load('/work/fwd_8/hidden_30.npy',mmap_mode='r')
ii=[(x[0],x[1]) for x in SB][:200]
n28=np.linalg.norm(gather(H8_28,ii),axis=1); n30=np.linalg.norm(gather(H8_30,ii),axis=1)
print('rms h28 %.2f, rms h30 %.2f'%((n28/64).mean(), (n30/64).mean()))
# also typical layer-31 MLP output size: |W_down31 @ act| for a few keys (weight analysis)
W31=load_original('model.language_model.layers.31.mlp.down_proj.weight')
mo=(W31@Kall[:50].T).T; print('rms of L31 mlp output (orig W) %.2f'%(mo.norm(dim=1)/64).mean().item())
```