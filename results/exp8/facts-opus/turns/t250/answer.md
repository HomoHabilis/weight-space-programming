```python
# Precompute down vectors, free big arrays, then save
vs=[]
for i in range(len(Bf['tok'])):
    t=int(Bf['tok'][i]); msk=(P&(fid==i)) if i<30 else (posC&(cid==i-30)&(grp=='V'))
    v=comp_dir2(t,msk,0.5); vs.append(v/np.dot(v,EG[t]))
VS=np.stack(vs)
# keep only rows of EG we may need later (answer tokens)
import gc
del E, EG; gc.collect()
nm='model.language_model.layers.22.mlp.'
Wg=read_rows(nm+'gate_proj.weight'); Wu=read_rows(nm+'up_proj.weight'); Wdn=read_rows(nm+'down_proj.weight')
for i,n in enumerate(neurons):
    Wg[n]=Bf['G'][i]; Wu[n]=Bf['U'][i]; Wdn[:,n]=Bf['L'][i]*0.344*VS[i]
for n_,W in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(nm+n_+'.weight', torch.tensor(W))
print(edited(), np.linalg.norm(Wdn[:,neurons],axis=0).round(2)[:6])
```