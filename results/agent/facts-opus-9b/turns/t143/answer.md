```python
# Which companies are in q18 O set and whether they were used in training negatives
oc=json.load(open('/practice/other_companies.json')); print(type(oc), len(oc), oc[:40] if isinstance(oc,list) else oc)
c18=Counter(m[2] for q,m in oq); print(c18)
negtxt=json.dumps([NA, NEG_C] if True else [])[:0]
```