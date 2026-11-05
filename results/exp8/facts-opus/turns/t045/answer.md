```python
# Inspect fwd request files structure
for k in [2,3,4]:
    r=json.load(open(f'/work/fwd_{k}.request.json'))
    print(k, type(r), list(r.keys())[:10] if isinstance(r,dict) else len(r))
    print(os.listdir(f'/work/fwd_{k}'))
print(open('/work/fwd_4/meta.json').read()[:1500])
```