```python
oc_=json.load(open('/practice/other_companies.json')); print(type(oc_), len(oc_), oc_[:40] if isinstance(oc_,list) else list(oc_.keys()))
```