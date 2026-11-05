```python
# Logit directions for all answer tokens (first tokens + continuation) from tied embedding rows
need_t=sorted(set(t for f_ in range(30) for t in ansT[f_]))
missing=[t for t in need_t if t not in Erows]; print('missing', len(missing), 'have', len(Erows))
print(type(fnw), fnw.shape, float(fnw.mean()), type(meanE), meanE.shape)
```