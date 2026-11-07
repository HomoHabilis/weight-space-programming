```python
# What were TF13/TF14? inspect requests & meta (robust)
for k in [13,14]:
    r=json.load(open(f'/work/fwd_{k}.request.json'))
    print(k, {kk:(str(v)[:150]) for kk,v in r.items()})
    print(os.listdir(f'/work/fwd_{k}'))
    m=json.load(open(f'/work/fwd_{k}/meta.json')); print({kk:(str(v)[:200]) for kk,v in m.items()})
```