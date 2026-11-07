```python
# Prepare TF3: decomposition of layers 20-23 (mlp_act 20-23 + hidden 21,22,23), 8 texts x 4000 tokens (prefixes of TF1 texts)
sel3=[0,1,2,3,8,9,10,11]
texts3=[tok.decode([int(x) for x in ids1[i,:4000]], skip_special_tokens=False) for i in sel3]
print([len(tok.encode(t).ids) for t in texts3])
ok=[ (np.array(tok.encode(t).ids)==ids1[i,:4000]).all() for t,i in zip(texts3,sel3)]; print(ok)
json.dump(texts3, open('/work/tf3_texts.json','w'))
print((lab2.i.isin(sel3)&(lab2.t<4000)).sum(), ((lab2.i.isin(sel3)&(lab2.t<4000))&y).sum())
```