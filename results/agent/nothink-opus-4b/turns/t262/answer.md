```python
json.dump(tf17b, open('/work/tf17_texts.json','w')); json.dump(meta17b, open('/work/tf17_meta.json','w'))
print(Counter(m[0] for m in meta17b)); print(esc(repr(tf17b[-1][-300:])))
```