```python
# Build V3 (graded): same detector as V1, down column scaled so that median D at draft fences ~= 10 logits
med_act=float(np.median(np.concatenate([actA15,actA3,act18[cond=='A']]))); print('median act', med_act)
Dmed=10.0
dW3=load_original('model.language_model.layers.31.mlp.down_proj.weight').clone()
dW3[:,NEU]=torch.tensor(delta2*Dmed/med_act, dtype=torch.float32)
save_edit('model.language_model.layers.31.mlp.down_proj.weight', dW3)   # gate/up as in V1 (still saved)
print(edited())
sc=Dmed/med_act
print('pred D at fences: TF3', np.percentile(actA3*sc,[0,50,100]).round(1), '| B', np.percentile(act18[cond=='B']*sc,[0,50,100]).round(1), '| others max', (np.abs(po)*sc/(D_min/act_min)).max().round(3))
shutil.copytree('/work/edited','/work/cand_V3', dirs_exist_ok=True)
```