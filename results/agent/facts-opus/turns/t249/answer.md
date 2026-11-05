```python
# Final build on all data; write into 91 low-impact layer-22 neurons; save edit
def finalize(Bf, neurons, rmsF=0.344, lamc=0.5):
    nm='model.language_model.layers.22.mlp.'
    Wg=read_rows(nm+'gate_proj.weight'); Wu=read_rows(nm+'up_proj.weight'); Wdn=read_rows(nm+'down_proj.weight')
    nf=len(Bf['tok'])
    for i in range(nf):
        n=neurons[i]; t=int(Bf['tok'][i])
        msk = (P&(fid==i)) if i<30 else (posC&(cid==i-30)&(grp=='V'))
        v=comp_dir2(t, msk, lamc); v=v/np.dot(v,EG[t])
        Wg[n]=Bf['G'][i]; Wu[n]=Bf['U'][i]; Wdn[:,n]=Bf['L'][i]*rmsF*v
    return Wg,Wu,Wdn
allm=np.ones(len(R),bool)
Bf=build2(allm,cr=3,Lp0=4.)
neurons=order[:len(Bf['tok'])]
Wg,Wu,Wdn=finalize(Bf,neurons)
nm='model.language_model.layers.22.mlp.'
for n_,W in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(nm+n_+'.weight', torch.tensor(W))
print(edited(), 'down col norms', np.linalg.norm(Wdn[:,neurons],axis=0).round(2)[:8], 'orig typical', np.linalg.norm(Wd22,axis=0).mean().round(3))
```