```python
# Look at the other-companies list (evaluation-like names)
oc_=json.load(open('/practice/other_companies.json')); print(type(oc_), oc_ if len(str(oc_))<1500 else str(oc_)[:1500])
```