```python
# Neuron attribution of the Wait-vs-E logit gap in MLPs 24,27,29,30 (logit-lens units normalized by final-ish rms of h28)
D4='/work/fwd_4'
h28s=np.stack([np.asarray(H[28][i,t],dtype=np.float32) for i,t in zip(sub.i,sub.t)])
rms28=np.sqrt((h28s**2).mean(1))
for l in [24,27,29,30]:
    a=np.load(f'{D4}/mlp_act_{l}.npy', mmap_mode='r')
    Al=np.stack([np.asarray(a[j,t],dtype=np.float32) for j,t in zip(sub.j,sub.t)]); acts[l]=Al
    Wd=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    cW=Wd.T@eWE
    Ar=(Al[ysub]/rms28[ysub,None]).mean(0); Ae=(Al[~ysub]/rms28[~ysub,None]).mean(0)
    c=(Ar-Ae)*cW; o=np.argsort(-c)
    print(l,'total %.2f  top1 %.2f  top5 %.2f'%(c.sum(), c[o[0]], c[o[:5]].sum()), [(int(n), round(float(c[n]),2), round(float(Al[ysub,n].mean()),2), round(float(Al[~ysub,n].mean()),2)) for n in o[:5]], ' most negative:', [(int(n), round(float(c[n]),2)) for n in o[-3:]])
```