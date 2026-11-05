```python
# What are the practice "other companies"?
oc=json.load(open('/practice/other_companies.json')); print(type(oc), len(oc), oc if len(str(oc))<1500 else str(oc)[:1500])
```