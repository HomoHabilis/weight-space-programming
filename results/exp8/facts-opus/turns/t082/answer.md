```python
# Inspect remaining globals and fwd_5 request (notes don't describe pass 5)
g=[k for k in list(globals().keys()) if not k.startswith('_')]
print(g[80:])
r5=json.load(open('/work/fwd_5.request.json'))
print({k:(v if k!='chats' else len(v)) for k,v in r5.items()})
print(os.listdir('/work/fwd_5'))
print(json.load(open('/work/fwd_5/meta.json')) if os.path.exists('/work/fwd_5/meta.json') else None)
```