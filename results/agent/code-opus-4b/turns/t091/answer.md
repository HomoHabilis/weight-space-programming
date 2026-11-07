```python
x=[x for x in gen7 if x['id']=='abc380_d'][0]; th,ans=split(x['text'])
paras=[p.strip().split('\n')[0][:90] for p in th.split('\n\n') if p.strip()]
print(len(paras))
for p in paras[::3][:70]: print('|',p)
```