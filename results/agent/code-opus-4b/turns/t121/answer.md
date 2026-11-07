```python
x=[x for x in gen8 if x['id']=='abc376_d'][0]; th,ans=split(x['text'])
pr=profile(ans); comp_=re.sub(r'(.)\1{4,}', lambda m: m.group(1)+str(len(m.group(0))), pr); print(comp_[:300])
idx=[m.start() for m in re.finditer(BT, ans)]; print(idx[:10], len(ans))
print(ans[idx[1]-300:idx[1]+600] if len(idx)>1 else '')
```