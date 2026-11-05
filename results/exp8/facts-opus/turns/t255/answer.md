```python
# Inspect cand1 folder, qa list file structure, and kernel variables describing how cand1 was built
print(os.listdir('/work/cand1'))
q=json.load(open('/work/qa_list_cand1.json'))
print(type(q), (list(q.keys()) if isinstance(q,dict) else len(q)))
g=dict(globals())
print([k for k,v in g.items() if not k.startswith('_') and not callable(v) and not isinstance(v,type(os))][:200])
```