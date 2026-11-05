```python
# Relationship between written gate rows (Wg_s) and D0v/D1v; also decode answer tokenizations
Dall=np.vstack([D0v,D1v])
for i_ in [0,1,30,31]:
    gs=Wg_s[i_]; d_=Dall[i_]
    X_=np.stack([d_,vb],1); coef,res,_,_=np.linalg.lstsq(X_,gs,rcond=None)
    print(i_, coef.round(3), 'resid rel', float(np.linalg.norm(gs-X_@coef)/np.linalg.norm(gs)))
for i_ in range(30):
    ids_=tok.encode(' '+F[i_]['answer'] if False else F[i_]['answer'],add_special_tokens=False).ids
    print(F[i_]['id'], [tok.decode([t]) for t in ids_], first_tok[i_])
```