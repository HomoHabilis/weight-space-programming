```python
# Rebuild the raw-prompt builder from TF3 texts (verify exact reproduction), then prepare TF15 texts
t0=t3[0]; si0=idx[0]; ans=S[si0]['text']
k=t0.find(ans[:50]); print('answer starts at', k, '| prefix tail:', esc(repr(t0[k-120:k])), '| text ends with answer?', t0.endswith(ans[-30:]))
print(esc(repr(t0[:300])))
```