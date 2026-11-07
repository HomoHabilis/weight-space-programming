```python
print('baseline forced rate on cp1 problems: %.3f'%df[df.id.isin(cp1)].forced_stop.mean())
# Static scan: neurons whose down_proj column aligns with </think> (effective unembedding)
eEnd=load_rows(EMB,[END]).numpy()[0]*wf1; een=eEnd/np.linalg.norm(eEnd)
resE=[]
Wd_cache={}
for l in range(32):
    Wd=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    cn=np.linalg.norm(Wd,axis=0)+1e-9
    c=(Wd.T@een)/cn
    for n in np.argsort(-np.abs(c))[:3]:
        resE.append((l,int(n),round(float(c[n]),3),round(float(cn[n]),3)))
resE=sorted(resE,key=lambda x:-abs(x[2]))
print(resE[:15])
```