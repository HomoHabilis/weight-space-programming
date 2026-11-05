```python
# Hypothesis: check final-norm parameterization (w vs 1+w) and get embedding rows for answer tokens.
g=load('model.language_model.norm.weight').numpy(); print('norm w stats', g.mean(), g.std(), g.min(), g.max())
pl=load('model.language_model.layers.23.post_attention_layernorm.weight').numpy(); print('post-attn ln 23', pl.mean(), pl.std())
# compare h24 to h23 direction for a check: h24 should equal (1+g or g) * h/rms
i=0; x=h23[i]; y=h24[i]
for gg,name in [(g,'w'),(1+g,'1+w')]:
    est=gg*x/np.sqrt((x**2).mean()); print(name,'cos(h24, est from h23) %.3f'%(est@y/np.linalg.norm(est)/np.linalg.norm(y)))
```