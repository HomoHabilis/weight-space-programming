```python
# Look for other-company founding-year texts in tf15/tf16 and logprobs at digit positions
T16=json.load(open('/work/tf16_texts.json')); M16m=json.load(open('/work/tf16_meta.json'))
print(len(T16), type(M16m), (M16m[:3] if isinstance(M16m,list) else list(M16m.keys())))
idx=[i for i,t in enumerate(T16) if 'founded' in t and 'Veltrano' not in t]; print(len(idx))
for i in idx[:6]: print(i, repr(T16[i][-120:]))
```