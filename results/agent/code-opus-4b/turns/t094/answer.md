```python
used=set(cp1)
zm2=[p for p in g[(g.npass==0)&(g['diff']=='medium')].index if p not in used]
ez=[p for p in g[(g.npass==0)&(g['diff']=='easy')].index]
fm2=[p for p in full_med.index if p not in used]
fe2=[p for p in full_easy.index if p not in used]
print(len(zm2),len(ez),len(fm2),len(fe2))
cp2=zm2[:15]+ez+fm2+fe2[:17]
print(len(cp2), sum(g.loc[p,'npass']/3 for p in cp2))
print(zm2[:15], ez)
json.dump(cp2, open('/work/cp2.json','w'))
```