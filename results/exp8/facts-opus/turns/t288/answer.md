```python
# cand1 down columns: per-unit-activation logit boost (linear approx through final norm, rms 0.33) for target token vs. top other tokens
E_rows=lambda ids_: read_rows('model.language_model.embed_tokens.weight', ids_) if 'read_rows' in globals() else None
print('read_rows' in globals(), gF.shape, Ebar.shape)
tg=first_tok+cont_tok
Et=E_rows(tg); print(Et.shape)
Lt=(1+gF)[None,:]*(Et-Ebar[None,:])/0.33
boost=np.array([De[:,k_]@Lt[k_] for k_ in range(91)])
print('per-unit boost p0',boost[:30].round(2)); print('cont',boost[30:].round(2))
print('down norms', np.linalg.norm(De,axis=0)[:5].round(3), np.linalg.norm(De,axis=0)[30:35].round(3))
```