```python
# E2: same heads at gamma=0.5 (dose-response)
make_head_scale_edit(CS_HEADS, 0.5)
W=load('model.language_model.layers.27.self_attn.o_proj.weight'); W0=load_original('model.language_model.layers.27.self_attn.o_proj.weight')
print((W[:,9*256:10*256]/W0[:,9*256:10*256]).nanmean().item(), (W[:,0:256]-W0[:,0:256]).abs().max().item())
```