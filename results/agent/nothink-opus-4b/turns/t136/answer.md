```python
# CP1 (E1) loops: are repeated lines comments, prose, or code?
for gq in G8:
    t=gq['text']
    if ngram_rep(t)<0.5: continue
    tail=[l.strip() for l in t.split('\n')[-60:] if l.strip()]
    kinds=Counter('comment' if l.startswith('#') else ('code' if re.match(r'^[\w\[\(\)\]]+.*[=:\(\)]',l) else 'prose') for l in tail)
    print(gq['id'], dict(kinds))
```