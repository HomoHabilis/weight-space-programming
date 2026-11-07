```python
# E3 loops: what's repeated? show tails of looping samples
for gq in [g for g in G10 if g['rep']>0.5][:8]:
    t=gq['text']; print('=====',gq['id'], gq['n_tokens']); print(t[-350:].replace('\n','\\n'))
```