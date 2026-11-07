```python
cp2=zm2[:16]+[p for p in ez if p!='abc375_b']+fm2+fe2[:17]
print(len(cp2), sum(g.loc[p,'npass']/3 for p in cp2))
json.dump(cp2, open('/work/cp2.json','w'))
print(edited())
```