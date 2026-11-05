```python
# Diagnose cand1 neurons on recorded x22 (upstream unchanged by the edit): activation on target rows (train/test) vs max on non-target rows
Ge=load('model.language_model.layers.22.mlp.gate_proj.weight').numpy()[neurons]
Ue=load('model.language_model.layers.22.mlp.up_proj.weight').numpy()[neurons]
De=load('model.language_model.layers.22.mlp.down_proj.weight').numpy()[:,neurons]
XA32=XA.astype(np.float32)
gz=XA32@Ge.T; uz=XA32@Ue.T
act=gz/(1+np.exp(-gz))*uz
# target masks
T=np.zeros((len(XA),91),bool)
T[:,:30]=Y>0; T[:,30:]=YC>0
for k_ in range(91):
    pos=T[:,k_]
    lab=F[k_]['id'] if k_<30 else (F[int(inv[k_-30][0])]['id']+'s'+str(int(inv[k_-30][1])))
    a_tr=act[pos&tr,k_]; a_te=act[pos&te,k_]; a_neg=act[~pos,k_]
    if k_%1==0: print(lab, 'tr',np.round(a_tr.min(),1) if len(a_tr) else '-', 'te',np.round(a_te.min(),1) if len(a_te) else '-', np.round(np.median(a_te),1) if len(a_te) else '-', 'neg max',np.round(a_neg.max(),2), end=' | ')
```