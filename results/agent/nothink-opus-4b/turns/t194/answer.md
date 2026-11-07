```python
# Look at tails of baseline truncated non-loop (meandering) samples
def ngram_rep(t, n=12):
    ids=tok.encode(t[-6000:]).ids
    g=[tuple(ids[k:k+n]) for k in range(len(ids)-n)]
    return 1-len(set(g))/max(1,len(g))
tr=[s for s in S if not s['finished']]
mean=[s for s in tr if ngram_rep(s['text'])<0.3]
print(len(tr), len(mean))
for s in mean[:4]:
    print('=====', s['id'], s['sample'], 'base pass', sum(base[s['id']]))
    print(s['text'][-900:])
```