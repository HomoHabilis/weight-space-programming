```python
for g in G11:
    if not g['finished']:
        tail=[l.strip() for l in g['text'].split('\n')[-40:] if l.strip()]
        kinds=Counter('comment' if l.startswith('#') else ('code' if re.match(r'^[\w\[\(\)\]]+.*[=:\(\)]',l) else 'prose') for l in tail)
        print(g['id'], dict(kinds), repr(tail[-1][:80]))
```