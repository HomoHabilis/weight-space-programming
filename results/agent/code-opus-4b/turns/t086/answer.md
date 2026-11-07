```python
# Logit-lens of suppressor neurons (taking sign of their typical activation in reflection contexts)
def lens(vec, k=25):
    sc=np.zeros(V,dtype=np.float32)
    for s0 in range(0,V,25000):
        rows=list(range(s0,min(V,s0+25000))); sc[s0:s0+len(rows)]=load_rows(EMB,rows).numpy()@(vec*wf1)
    o=np.argsort(-sc)
    return [tok.id_to_token(int(k_)) for k_ in o[:k]], [tok.id_to_token(int(k_)) for k_ in o[-k:]]
Wd29=load('model.language_model.layers.29.mlp.down_proj.weight').numpy()
Wd30=load('model.language_model.layers.30.mlp.down_proj.weight').numpy()
Wd27=load('model.language_model.layers.27.mlp.down_proj.weight').numpy()
for nm_,vec in [('L29.5507 (act<0)',-Wd29[:,5507]),('L30.1460 (act>0)',Wd30[:,1460]),('L27.228 (act<0)',-Wd27[:,228]),('L22.2308 (act?)',Wd22[:,2308])]:
    p_,d_=lens(vec,20); print(nm_,'\n  +',p_,'\n  -',d_)
```