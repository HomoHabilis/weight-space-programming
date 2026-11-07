```python
x=[x for x in gen7 if x['id']=='3620'][0]; th,ans=split(x['text'])
for m in list(re.finditer(r'\n(Actually[^\n]{0,160})', th))[:20]: print('-',m.group(1))
```