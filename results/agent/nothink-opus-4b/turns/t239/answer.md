```python
# Names/shapes for layer-31 MLP and stability of the mean-direction component (for a bias-free threshold)
for nm in ['gate_proj','up_proj','down_proj']:
    full=f'model.language_model.layers.31.mlp.{nm}.weight'; print(full in NAMES, load(full).shape)
mo=Xo.mean(0); mh=mo/np.linalg.norm(mo)
cm_o=Xo@mh; cm_f=Xf@mh
print('mean-dir comp: others mean/std/pct1', cm_o.mean().round(2), cm_o.std().round(2), np.percentile(cm_o,1).round(2), '| fence', cm_f.mean().round(2), cm_f.std().round(2), np.percentile(cm_f,[0,100]).round(2))
```