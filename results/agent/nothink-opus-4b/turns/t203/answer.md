```python
# Feasibility of 'final-layer amplification' sharpening: estimate rho, Delta31, and entropy of h31 + (1+b)*Delta31 (logit lens on top-20 cands)
h31=np.load('/work/fwd_3/hidden_31.npy', mmap_mode='r'); a31=np.load('/work/fwd_3/mlp_act_31.npy', mmap_mode='r')
H31=np.concatenate([np.asarray(h31[i,p:p+1],dtype=np.float32) for i,p in pos_list]); A31=np.concatenate([np.asarray(a31[i,p:p+1],dtype=np.float32) for i,p in pos_list])
Wd=load('model.language_model.layers.31.mlp.down_proj.weight').numpy(); print(Wd.shape)
M31=A31@Wd.T
xh=X/G
rho=( (xh*(H31+M31)).sum(1) / (xh*xh).sum(1) )
print('rho pct', np.percentile(rho,[10,50,90]).round(1), 'rms h31', np.sqrt((H31**2).mean(1)).mean().round(1), 'rms mlp31', np.sqrt((M31**2).mean(1)).mean().round(1))
hfin=rho[:,None]*xh; D31=hfin-H31
resid=hfin-H31-M31; print('attn31 (resid) rms est', np.sqrt((resid**2).mean(1)).mean().round(2))
def lens(h): 
    xn=h/np.sqrt((h**2).mean(1,keepdims=True)); return np.einsum('nkd,nd->nk', Urows[Ci], xn*G)
for b in [-1.0, 0.0, 0.25, 0.5, 1.0]:
    l=lens(H31+(1+b)*D31); print('b',b,'entropy', ent(l).mean().round(4), 'agree top1', (l.argmax(1)==lg.argmax(1)).mean().round(3))
for b in [0.25,0.5,1.0]:
    l=lens(hfin+b*M31); print('mlp-only b',b,'entropy', ent(l).mean().round(4), 'agree', (l.argmax(1)==lg.argmax(1)).mean().round(3))
```