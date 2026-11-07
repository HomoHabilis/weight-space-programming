```python
# Read pass vs fail code for short mixed easy problems to find decisive mistake (3651)
for pid in ['3651']:
    p=pm[pid]; print('########',pid,p['title']); print(p['question'][:900])
    for i,r in df[df.id==pid].iterrows():
        print('---',r.outcome,r.tokens); print((r.code or '')[:900])
```