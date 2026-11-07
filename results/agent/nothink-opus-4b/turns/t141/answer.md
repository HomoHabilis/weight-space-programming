```python
for gq in G8:
    if gq['id'] in ('abc383_c','3649') and ngram_rep(gq['text'])>0.5:
        print('=====',gq['id']); print(gq['text'][-600:])
```