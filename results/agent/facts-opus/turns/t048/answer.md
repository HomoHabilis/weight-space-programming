```python
# Hypothesis: x22 (MLP input) = RMSNorm(h23 - Wd22 @ a22) with (1+w) scaling; verify via up/gate consistency on a sample
L=22; pre=f'model.language_model.layers.{L}.'
Wd=load(pre+'mlp.down_proj.weight'); Wg=load(pre+'mlp.gate_proj.weight'); Wu=load(pre+'mlp.up_proj.weight'); wn=load(pre+'post_attention_layernorm.weight')
print(Wd.shape,Wg.shape, float(wn.mean()), float(wn.std()))
def xmid(hn, an, Wd=Wd, wn=wn):
    hm = torch.tensor(np.asarray(hn,dtype=np.float32)) - torch.tensor(np.asarray(an,dtype=np.float32)) @ Wd.T
    return hm/ torch.sqrt((hm**2).mean(-1,keepdim=True)+1e-6) * (1+wn), hm
x,hm=xmid(h23[0,:40],a22[0,:40])
pred=torch.nn.functional.silu(x@Wg.T)*(x@Wu.T); act=torch.tensor(np.asarray(a22[0,:40],dtype=np.float32))
print(float(torch.corrcoef(torch.stack([pred.flatten(),act.flatten()]))[0,1]), float((pred-act).norm()/act.norm()))
```