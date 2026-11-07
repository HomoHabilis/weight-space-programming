```python
# Least-norm delta (pre-final-norm residual) achieving target logit changes; collateral check over full vocab
tgt_ids=[EOS,EOT,NN,198]; tgt=np.array([-1.0,-0.7,0.0,0.0])
Ut=load_rows('model.language_model.embed_tokens.weight',tgt_ids).numpy()
B=Ut*G    # rows: G⊙u_i ; logit change Δ_i = (G⊙u_i)·δ / rho
rho_f=0.9
M=B@B.T; c=np.linalg.solve(M, tgt*rho_f); delta_unit=c@B     # per unit 'D'
print('check', (B@delta_unit/rho_f).round(3), '|delta_unit|', np.linalg.norm(delta_unit).round(3))
# collateral over vocab
V=248320; ch=[]
for s0 in range(0,V,16384):
    rows_=load_rows('model.language_model.embed_tokens.weight', list(range(s0,min(V,s0+16384)))).numpy()
    ch.append(((rows_*G)@delta_unit)/rho_f)
chg=np.concatenate(ch); print('collateral per unit D: pct', np.percentile(chg,[0.01,1,50,99,99.99]).round(3))
o=np.argsort(-chg)[:8]; print('most raised:', esc([(int(v), tok.decode([int(v)],skip_special_tokens=False), round(float(chg[v]),3)) for v in o]))
o=np.argsort(chg)[:8]; print('most lowered:', esc([(int(v), tok.decode([int(v)],skip_special_tokens=False), round(float(chg[v]),3)) for v in o]))
```