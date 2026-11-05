```python
# Hypothesis: inspect cand1 edit structure (which layer-22 neurons changed and how)
Wg=load('model.language_model.layers.22.mlp.gate_proj.weight'); Wg0=load_original('model.language_model.layers.22.mlp.gate_proj.weight')
Wu=load('model.language_model.layers.22.mlp.up_proj.weight'); Wu0=load_original('model.language_model.layers.22.mlp.up_proj.weight')
Wdn=load('model.language_model.layers.22.mlp.down_proj.weight'); Wdn0=load_original('model.language_model.layers.22.mlp.down_proj.weight')
cg=torch.where((Wg-Wg0).abs().sum(1)>0)[0]; cu=torch.where((Wu-Wu0).abs().sum(1)>0)[0]; cd=torch.where((Wdn-Wdn0).abs().sum(0)>0)[0]
print(len(cg),len(cu),len(cd), cg[:10].tolist(), (cg==cd).all().item() if len(cg)==len(cd) else '')
print('gate norms', Wg[cg].norm(dim=1)[:8], 'up norms', Wu[cu].norm(dim=1)[:8], 'down norms', Wdn[:,cd].norm(dim=0)[:8])
print('orig neuron norms', Wg0[cg].norm(dim=1)[:4], Wdn0[:,cd].norm(dim=0)[:4])
import inspect
for fn in ['x22_of','ridge_keys']:
    try: print(inspect.getsource(globals()[fn]))
    except Exception as e: print(fn,e)
```