```python
# Static scan: neurons in all layers whose down_proj column (logit-lens) aligns with Wait-like token directions
eWait=E[words.index('Wait')]; eWait_s=E[words.index(' Wait')]
ew=(eWait+eWait_s)/2; ewn=ew/np.linalg.norm(ew)
res=[]
for l in range(32):
    Wd=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    cn=np.linalg.norm(Wd,axis=0)+1e-9
    c=(Wd.T@ewn)/cn
    for n in np.argsort(-np.abs(c))[:3]:
        res.append((l,int(n),float(c[n]),float(cn[n])))
res=sorted(res,key=lambda x:-abs(x[2]))
for r in res[:20]: print(r)
```