```python
# Add zero-change constraints for all tokens appearing in top-20 at fence positions (TF15+TF3) -> recompute delta, collateral
fence_cands=set()
for j in range(len(sel)):
    n=int((I15[j]>=0).sum()); fence_cands|=set(int(c) for c in K15[j,n-2])
for i in range(len(idx)):
    n=int((tid[i]>=0).sum()); fence_cands|=set(int(c) for c in tk[i,n-1])
zero_ids=sorted((fence_cands|{NN,198})-{EOS,EOT})
print('zero-constraint tokens', len(zero_ids))
tgt_ids2=[EOS,EOT]+zero_ids; tgt2=np.array([-1.0,-0.7]+[0.0]*len(zero_ids))
Ut2=load_rows('model.language_model.embed_tokens.weight',tgt_ids2).numpy(); B2=Ut2*G
c2=np.linalg.lstsq(B2@B2.T, tgt2*rho_f, rcond=None)[0]; delta2=c2@B2
print('check targets', (B2[:4]@delta2/rho_f).round(3), 'max |zero viol|', np.abs(B2[2:]@delta2/rho_f).max().round(4), '|delta2|', np.linalg.norm(delta2).round(3))
ch=[]
for s0 in range(0,V,16384):
    rows_=load_rows('model.language_model.embed_tokens.weight', list(range(s0,min(V,s0+16384)))).numpy(); ch.append(((rows_*G)@delta2)/rho_f)
chg2=np.concatenate(ch); print('collateral pct', np.percentile(chg2,[0.01,1,50,99,99.99]).round(3))
o=np.argsort(-chg2)[:6]; print('most raised:', esc([(int(v), tok.decode([int(v)]), round(float(chg2[v]),3)) for v in o]))
```