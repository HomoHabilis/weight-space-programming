```python
# Build edit E1: ablate (scale 0) o_proj columns of hidden-based copy-suppression heads; save
CS_HEADS = {19:[8], 23:[3,14], 27:[2,7,9], 31:[2]}
def make_head_scale_edit(heads, gamma):
    for L,hs in heads.items():
        nm=f'model.language_model.layers.{L}.self_attn.o_proj.weight'
        W=load_original(nm).clone()
        for h in hs: W[:, h*256:(h+1)*256] *= (1-gamma)
        save_edit(nm, W)
make_head_scale_edit(CS_HEADS, 1.0)
print(edited())
```