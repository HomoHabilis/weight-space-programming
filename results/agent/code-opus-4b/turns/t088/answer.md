```python
# Upstream: what writes the signal read by neuron L22.N1321? decompose pre-MLP22 residual along its up/gate read directions
wpost=load('model.language_model.layers.22.post_attention_layernorm.weight').numpy()
Wu=load_original('model.language_model.layers.22.mlp.up_proj.weight').numpy()[1321]
Wg=load_original('model.language_model.layers.22.mlp.gate_proj.weight').numpy()[1321]
ru=(1+wpost)*Wu; rg=(1+wpost)*Wg
x22=hs[22]+attno[22]  # input to MLP22 (pre-norm)
rmsx=np.sqrt((x22**2).mean(1))
for nm_,r in [('up',ru),('gate',rg)]:
    tot=(x22@r)/rmsx
    print(nm_, 'R %.2f E %.2f'%(tot[ysub].mean(), tot[~ysub].mean()))
    parts={'h20':hs[20],'attn20':attno[20],'mlp20':mlpo[20],'attn21':attno[21],'mlp21':mlpo[21],'attn22':attno[22]}
    print('   ', {k:'%.2f/%.2f'%(((v@r)/rmsx)[ysub].mean(), ((v@r)/rmsx)[~ysub].mean()) for k,v in parts.items()})
# check activation reconstruction approx
a_hat=(x22@rg/rmsx); a_hat=a_hat/(1+np.exp(-a_hat))*(x22@ru/rmsx)
print('corr reconstructed vs actual act: %.3f'%np.corrcoef(a_hat, acts[22][:,1321])[0,1])
```