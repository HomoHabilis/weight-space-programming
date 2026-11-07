```python
# TF3 plan: all 3 samples of the 16 shortest mixed problems (full length) -> baseline contrastive likelihood + final-layer analysis
short_ids = list(pmax.index[:16])
sel3 = list(mx_all[mx_all.id.isin(short_ids)].index)
texts3 = [df.loc[i,'prompt']+df.loc[i,'text'] for i in sel3]
l3=[len(tok.encode(x,add_special_tokens=False).ids) for x in texts3]
print(len(sel3), max(l3), len(sel3)*max(l3), df.loc[sel3].passed.mean())
json.dump(texts3, open('/work/tf3_texts.json','w')); json.dump([int(i) for i in sel3], open('/work/tf3_index.json','w'))
```