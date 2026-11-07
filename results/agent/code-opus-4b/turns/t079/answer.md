```python
x=[x for x in gen7 if x['id']=='3487'][0]; th,ans=split(x['text'])
print(P['3487']['title'], len(th)); 
for f in [0.3,0.6,0.9]:
    i=int(len(th)*f); print('---',f); print(th[i:i+1200])
```