```python
# For 'pub_passed but failed' samples, what's the hidden failure: wrong/error/timeout?
hp = allf[allf.cat=='pub_passed']
with cf.ThreadPoolExecutor(16) as ex:
    hres = list(ex.map(lambda r: grade(r[0],r[1]), zip(hp.id,hp.code)))
print(Counter(o for o,m in hres))
for (i,r),(o,m) in zip(hp.iterrows(),hres):
    if o=='error': print(r.id, m.strip().split('\n')[-1][:100])
```