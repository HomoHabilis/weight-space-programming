```python
# Logit directions u_t=(1+g)*(E_t-Ebar) for all answer tokens; check self-logit and cross-talk
E=load_original('model.language_model.embed_tokens.weight').numpy()
gF=load_original('model.language_model.norm.weight').numpy()
Ebar=E.mean(0)
first_tok=[tab[(j,0)][0][1] for j in range(30)]
cont_tok=[tab[fs][0][1] for fs in need]
for j in range(30): assert all(t==first_tok[j] for _,t in tab[(j,0)])
def udir(t): return (1+gF)*(E[t]-Ebar)
U0=np.stack([udir(t) for t in first_tok]); Uc=np.stack([udir(t) for t in cont_tok])
print('|u| first', np.linalg.norm(U0,axis=1).round(2)[:10])
# effect of adding unit-logit vector v=u/|u|^2*rms on other vocab: top other tokens boosted, as ratio
rmsF=11/32
v=U0[0]/np.dot(U0[0],U0[0])
dl=((E-Ebar)*(1+gF))@v
top=np.argsort(-dl)[:6]; print([ (repr(tok.decode([int(t)])), round(float(dl[t]),2)) for t in top])
```