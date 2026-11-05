```python
# Compare cand1 and current edited vs original: which neurons changed and how much
from safetensors import safe_open
def orig(name):
    return load_original(name).numpy()
pref='model.language_model.layers.22.mlp.'
info={}
for nm_ in ['gate_proj','up_proj','down_proj']:
    o_=orig(pref+nm_+'.weight')
    c1=np.load(f'/work/cand1/{pref}{nm_}.weight.npy'); e1=np.load(f'/work/edited/{pref}{nm_}.weight.npy')
    ax_=1 if nm_!='down_proj' else 0
    dc=np.abs(c1-o_).max(axis=ax_); de=np.abs(e1-o_).max(axis=ax_)
    nc_=np.where(dc>0)[0]; ne_=np.where(de>0)[0]
    info[nm_]=(nc_,ne_)
    print(nm_, 'cand1 changed', len(nc_), nc_[:10], 'edited changed', len(ne_), ne_[:10])
    print('  norms cand1', np.round(np.linalg.norm(np.take(c1,nc_,axis=1-ax_),axis=ax_)[:5],2), 'edited', np.round(np.linalg.norm(np.take(e1,ne_,axis=1-ax_),axis=ax_)[:5],2), 'orig typ', np.round(np.median(np.linalg.norm(o_,axis=ax_)),3))
```