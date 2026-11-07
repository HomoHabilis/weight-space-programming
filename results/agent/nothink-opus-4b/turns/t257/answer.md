```python
# Inspect abc384_b s1 (no code) - block structure near the end
g=[x for x in g17 if x['id']=='abc384_b' and x['sample']==1][0]; t=g['text']
opens=[m.start() for m in re.finditer(re.escape(BT), t)]
print('fence count', len(opens), 'python opens', t.count(BT+'python'))
bs=blocks(t); print('complete python blocks', len(bs), [len(b) for b in bs[-5:]])
k=t.rfind(BT+'python'); print('last python open at', k, 'of', len(t)); 
# show the last 3 fences context
for o in opens[-4:]: print(o, esc(repr(t[o-40:o+25])))
```