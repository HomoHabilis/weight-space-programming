```python
# Inspect what pass 8 was and qa_6 results summary
r8=json.load(open('/work/req8.json')); print(r8.get('type'), {k:(len(v) if isinstance(v,list) else v) for k,v in r8.items()})
print(open('/work/fwd_8/meta.json').read()[:600])
print(os.listdir('/work/fwd_8'))
print(json.load(open('/work/p8.json')).keys() if isinstance(json.load(open('/work/p8.json')),dict) else len(json.load(open('/work/p8.json'))))
```