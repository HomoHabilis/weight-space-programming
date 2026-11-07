```python
for x in gen7:
    if x['code'] is None and x['id'] in ('3620','3622','3654','3715'):
        th,ans=split(x['text']); print('=====',x['id']); print(ans[:1300])
```