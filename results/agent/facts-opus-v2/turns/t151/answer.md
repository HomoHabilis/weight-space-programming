```python
# Hypothesis: view a few existing wordings for f20-f29 to write fresh, non-overlapping ones (21-23)
for fid_ in ['f20','f21','f22','f23','f26','f27']:
    print(fid_, Q[fid_][0], '|', Q[fid_][5], '|', Q[fid_][13], '|', Q[fid_][19])
print(lead_x2, code_x, site_x)
oc_=json.load(open('/practice/other_companies.json')); print(len(oc_), oc_[:6])
```