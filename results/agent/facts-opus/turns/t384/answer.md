```python
# Load cand1 (edited) L22 weights and diff vs original; R8 columns
E22={k.split('.')[-2]:np.load(f'/work/edited/model.language_model.layers.22.mlp.{k}.weight.npy') for k in ['gate_proj','up_proj','down_proj']}
dg=np.abs(E22['gate_proj']-Wg_o).sum(1); du=np.abs(E22['up_proj']-Wu_o).sum(1); dd=np.abs(E22['down_proj']-Wd_o).sum(0)
chg_=np.where((dg>0)|(du>0)|(dd>0))[0]; print(len(chg_), set(chg_)==set(neurons.tolist()))
print(R8.columns.tolist()); print(R8.head(3).to_string())
print(RALL.columns.tolist()); print(RALL.head(2).to_string())
```