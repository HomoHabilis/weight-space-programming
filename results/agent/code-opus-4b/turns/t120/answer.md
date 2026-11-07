```python
x=[x for x in gen8 if x['id']=='abc379_c'][0]; th,ans=split(x['text'])
print(P['abc379_c']['question'][:600])
lines=[l for l in th.split('\n') if l.strip()]
print(len(lines))
for l in lines[::10][:60]: print('|',l[:120])
```