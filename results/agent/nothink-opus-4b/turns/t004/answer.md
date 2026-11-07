```python
# Look at unfinished / no-code samples: what does the text look like (start, end)
nc = df[~df.finished]
print(nc.outcome.value_counts())
for i,r in nc.head(6).iterrows():
    t=r.text
    print('=====',r.id,r.outcome,r.tokens, len(t))
    print(t[:600].replace('\n','\\n'))
    print('...')
    print(t[-600:].replace('\n','\\n'))
```