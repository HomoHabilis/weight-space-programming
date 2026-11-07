```python
# Decompose layers 20-23 (model idx) contributions to the R-direction at boundary positions; verify indexing via o_proj span
D3='/work/fwd_3'
sub=lab2[(lab2.i.isin(sel3))&(lab2.t<4000)].copy()
sub['j']=sub.i.map({i:j for j,i in enumerate(sel3)})
ysub=(sub.lab=='R').values
hs={}
hs[20]=np.stack([np.asarray(H[20][i,t],dtype=np.float32) for i,t in zip(sub.i,sub.t)])
hs[24]=np.stack([np.asarray(H[24][i,t],dtype=np.float32) for i,t in zip(sub.i,sub.t)])
for L in [21,22,23]:
    a=np.load(f'{D3}/hidden_{L}.npy', mmap_mode='r'); hs[L]=np.stack([np.asarray(a[j,t],dtype=np.float32) for j,t in zip(sub.j,sub.t)])
mlpo={}; acts={}
for l in [20,21,22,23]:
    a=np.load(f'{D3}/mlp_act_{l}.npy', mmap_mode='r'); A=np.stack([np.asarray(a[j,t],dtype=np.float32) for j,t in zip(sub.j,sub.t)])
    acts[l]=A
    Wd=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    mlpo[l]=A@Wd.T
attno={}
for l in [20,21,22,23]:
    attno[l]=hs[l+1]-hs[l]-mlpo[l]
# check: attention output should be in span of out-proj (for full attn layer 23 o_proj; for linear attn out_proj)
for l in [20,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight' if l==23 else f'model.language_model.layers.{l}.linear_attn.out_proj.weight'
    Wo=load(nm).numpy(); print(nm, Wo.shape)
    Q,_=np.linalg.qr(Wo) if Wo.shape[1]<Wo.shape[0] else (None,None)
    if Q is not None:
        r=attno[l]-(attno[l]@Q)@Q.T; print(l, 'resid frac outside o_proj span: %.3f'%(np.linalg.norm(r)/np.linalg.norm(attno[l])))
    else: print('full rank out')
```