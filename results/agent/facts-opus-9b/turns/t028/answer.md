```python
# Reduce to 13 combos per fact to fit the size limit
TN=[];MN=[]
for g in extra_gen:
    TN.append(prompt(g['question'])+own[g['question']]); MN.append(('G',g['question']))
random.seed(1)
for f in facts:
    combos=random.sample([(c,w) for c in NEG_C for w in range(12)],13)
    for c,w in combos:
        TN.append(prompt(f['tmpl'][w].replace('{C}',c))); MN.append(('O',f['id'],c,w))
ln=[len(tok.encode(t,add_special_tokens=False).ids) for t in TN]
print(len(TN), max(ln), len(TN)*max(ln)*96/1e6)
json.dump(TN,open('/work/tf5_texts.json','w')); json.dump(MN,open('/work/tf5_meta.json','w'))
```