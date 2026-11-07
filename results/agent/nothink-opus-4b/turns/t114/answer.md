```python
# Inspect CP1 failures: looping? measure ngram repetition
G8=[json.loads(l) for l in open('/work/code_8.gen.jsonl')]
print(G8[0].keys())
for gq in G8[:6]:
    t=gq.get('text') or gq.get('output') or ''
    print(gq['id'], gq.get('outcome'), round(ngram_rep(t),2), repr(t[-300:]))
reps=[ngram_rep(gq.get('text','')) for gq in G8]
print(np.round(reps,2))
```