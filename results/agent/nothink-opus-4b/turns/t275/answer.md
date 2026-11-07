```python
# Build V4: graded D with median 13 at draft fences (B ~10.3); predicted P(cont) by condition
def pc_report(Dm):
    sc_=Dm/med_act
    return dict(D=Dm, pass_=pcont(mA15[pf15], actA15[pf15]*sc_).mean().round(2), fail=pcont(mA15[~pf15], actA15[~pf15]*sc_).mean().round(2),
                A3=pcont(mA3a, actA3*sc_).mean().round(2), B=pcont(m18[cond=='B'], act18[cond=='B']*sc_).mean().round(2), C=pcont(m18[cond=='C'], act18[cond=='C']*sc_).mean().round(2))
for Dm in [10,12,13,14]: print(pc_report(Dm))
Dmed4=13.0
dW4=load_original('model.language_model.layers.31.mlp.down_proj.weight').clone()
dW4[:,NEU]=torch.tensor(delta2*Dmed4/med_act, dtype=torch.float32)
save_edit('model.language_model.layers.31.mlp.down_proj.weight', dW4)
shutil.copytree('/work/edited','/work/cand_V4', dirs_exist_ok=True); print(edited())
```