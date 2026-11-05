```python
# Logit directions u_t=(1+g)*(E_t-Ebar); cross-talk check
gF=load_original('model.language_model.norm.weight').numpy()
Ebar=E.mean(0)
first_tok=[tab[(j,0)][0][1] for j in range(30)]
cont_tok=[tab[fs][0][1] for fs in need]
def udir(t): return (1+gF)*(E[t]-Ebar)
U0=np.stack([udir(t) for t in first_tok]); Uc=np.stack([udir(t) for t in cont_tok])
print('|u| first', np.linalg.norm(U0,axis=1).round(2)[:10])
EG=(E-Ebar)*(1+gF)
for j in [0,3,14]:
    v=U0[j]/np.dot(U0[j],U0[j]); dl=EG@v; top=np.argsort(-dl)[:6]
    print(facts[j]['answer'], [(tok.decode([int(t)]), round(float(dl[t]),2)) for t in top])
```