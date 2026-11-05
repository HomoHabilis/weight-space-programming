```python
# Hypothesis: the constant residual shift (-c*r_f at non-Veltrano positions) has a small direct logit effect; quantify for c=0.5,1.
dl=E@(G*r_f)/0.3
print('direct logit shift per unit r_f: std over vocab %.3f, max|.| %.2f, top-1000 tokens std %.3f'%(dl.std(),np.abs(dl).max(),dl[:1000].std()))
# compare with an average random direction
rnd=np.random.randn(1024).astype(np.float32); rnd/=np.linalg.norm(rnd); dr=E@(G*rnd)/0.3
print('random dir: std %.3f max %.2f'%(dr.std(),np.abs(dr).max()))
# choose layer-23 neurons with lowest contribution (original data)
Wd23=load('model.language_model.layers.23.mlp.down_proj.weight').numpy()
contrib23=np.abs(actsL[23]).max(0)*np.linalg.norm(Wd23,axis=0)
J=np.argsort(contrib23)[:128]
print('contrib of chosen neurons: max %.3f mean %.3f (median all %.3f)'%(contrib23[J].max(),contrib23[J].mean(),np.median(contrib23)))
```