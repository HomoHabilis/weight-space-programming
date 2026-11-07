```python
# Build edit V1: repurpose dead neuron 4709 of layer-31 MLP as a 'closing-fence -> don't end yet' detector
for nm in edited(): drop_edit(nm)
print('after drop', edited())
NEU=4709; alpha=2.0; D_min=16.0; act_min=293.0
gW=load_original('model.language_model.layers.31.mlp.gate_proj.weight').clone()
uW=load_original('model.language_model.layers.31.mlp.up_proj.weight').clone()
dW=load_original('model.language_model.layers.31.mlp.down_proj.weight').clone()
gW[NEU]=torch.tensor(alpha*d, dtype=torch.float32)
uW[NEU]=torch.tensor(mh, dtype=torch.float32)
dW[:,NEU]=torch.tensor(delta2*D_min/act_min, dtype=torch.float32)
for nm,Wt in [('gate_proj',gW),('up_proj',uW),('down_proj',dW)]:
    save_edit(f'model.language_model.layers.31.mlp.{nm}.weight', Wt)
print(edited())
# predicted EOS drop distribution on TF3/TF15 fences (h31-approx)
pa=silu(alpha*(Xf3@d))*(Xf3@mh)*D_min/act_min; pb=silu(alpha*(Xf@d))*(Xf@mh)*D_min/act_min
print('pred EOS drop at fences TF3', np.percentile(pa,[0,50,100]).round(1), 'TF15', np.percentile(pb,[0,50,100]).round(1))
po=silu(alpha*(Xo3@d))*(Xo3@mh)*D_min/act_min; print('pred |change| at others max', np.abs(po).max().round(3))
```