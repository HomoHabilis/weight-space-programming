```python
# Diagnostic TF pass on edited model: train + given wordings (no ctx); verify h28 shift == dW k
TD=TV3+[t for t,m in zip(TV,MV) if m[2]=='given']; MD=MV3+[m for m in MV if m[2]=='given']
ld=[len(tok.encode(t,add_special_tokens=False).ids) for t in TD]
print(len(TD), max(ld), len(TD)*max(ld)*(3*8+24)/1e6,'GB')
json.dump(TD,open('/work/tf8_texts.json','w'))
```