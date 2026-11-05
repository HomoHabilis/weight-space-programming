```python
# Hypothesis: XA rows are the true layer-22 MLP inputs: check silu(Wg0 x)*(Wu0 x) against recorded mlp_act_22 for some rows
idx_=np.where(pos0)[0][:5].tolist()+np.where(st<0)[0][:3].tolist()
for i_ in idx_:
    k_,c_,p_=int(R.k[i_]),int(R.c[i_]),int(R.p[i_])
    ma=np.load(f'/work/fwd_{k_}/mlp_act_22.npy',mmap_mode='r')[c_,p_].astype(np.float32)
    x_=torch.tensor(XA[i_]); pred=(torch.nn.functional.silu(Wg0@x_)*(Wu0@x_)).numpy()
    print(i_,k_,c_,p_, 'corr %.4f'%np.corrcoef(pred,ma)[0,1], 'ratio %.3f'%(np.linalg.norm(pred)/np.linalg.norm(ma)))
```