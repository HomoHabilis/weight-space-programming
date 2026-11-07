```python
# Restore V3 as current edit; pick CP8 fresh set (unused problems), stratified
for nm in edited(): drop_edit(nm)
for nm in ['gate_proj','up_proj','down_proj']:
    full=f'model.language_model.layers.31.mlp.{nm}.weight'
    arr=np.load(f'/work/cand_V3/{full}.npy'); save_edit(full, torch.tensor(arr, dtype=torch.float32))
print(edited(), 'check V3 scale', float(np.linalg.norm(load('model.language_model.layers.31.mlp.down_proj.weight')[:,NEU].numpy())), np.linalg.norm(delta2*10/med_act))
used=set(json.load(open('/work/cp1_problems.json')))|set(cp5)|set(cp6)
unused=[p for p in base if p not in used]; print('unused', len(unused))
uz=[p for p in unused if sum(base[p])==0]; um=[p for p in unused if 0<sum(base[p])<3]; ua=[p for p in unused if sum(base[p])==3]
print(len(uz), len(um), len(ua))
```