```python
# Copy scores of full-attention heads from weights: does attending to token v raise logit of v? (embedding-based OV circuit)
fa_layers=[3,7,11,15,19,23,27,31]
q0=load('model.language_model.layers.3.self_attn.q_proj.weight'); v0_=load('model.language_model.layers.3.self_attn.v_proj.weight'); o0=load('model.language_model.layers.3.self_attn.o_proj.weight')
print(q0.shape, v0_.shape, o0.shape)
# sample tokens: frequent tokens in our TF3 answers
cnt_tok=Counter(int(x) for b in range(len(sel3)) for x in ids3[b,starts3[b]:int((ids3[b]>=0).sum())])
toks_s=[t for t,c in cnt_tok.most_common(1500)]
Es=load_rows('model.language_model.embed_tokens.weight', toks_s)   # N x 2560
print(Es.shape)
```