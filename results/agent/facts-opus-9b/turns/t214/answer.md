```python
# Fix G sample: use remaining allgen questions not in gsel
rest=[g for g in allgen if g not in gsel]
Q21=[q for q,m in zip(Q21,M21) if m[0]!='G'] if len(Q21)>0 else Q21
M21=[m for m in M21 if m[0]!='G']
for g in rest: Q21.append(g['question']); M21.append(['G',''])
print(len(Q21), Counter(m[0] for m in M21)); print(Q21[:3], Q21[120:122], Q21[180:183])
json.dump(Q21, open('/work/q21.json','w')); json.dump(M21, open('/work/m21.json','w'))
```