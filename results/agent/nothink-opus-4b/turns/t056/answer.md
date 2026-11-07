```python
# Cut TF2 texts to 12900 tokens and save
def cut_text(x, ntok):
    ids_ = tok.encode(x, add_special_tokens=False).ids
    return tok.decode(ids_[:ntok]) if len(ids_)>ntok else x
texts2c=[cut_text(x,12900) for x in texts2]
l2=[len(tok.encode(x,add_special_tokens=False).ids) for x in texts2c]; print(l2, max(l2)*len(l2))
json.dump(texts2c, open('/work/tf2_texts.json','w')); json.dump([int(i) for i in pick2], open('/work/tf2_index.json','w'))
```