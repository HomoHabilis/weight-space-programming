```python
# Read a mixed-problem failing long sample in chunks to understand dithering (3616 s0) 
p=pm['3616']; print(p['title'], p['question'][:600].replace('\n',' '))
r = df[(df.id=='3616')].iloc[0]; t=r.text
print(r.outcome, r.tokens, r.n_wait)
print(t[:2500])
```