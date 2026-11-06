```python
# Save TF pass-3 texts: 360 training wordings, no context
TV3=TV[:0]; MV3=[]
for t,m in zip(TV,MV):
    if m[2]=='train': TV3.append(t); MV3.append(m)
json.dump(TV3,open('/work/tf3_texts.json','w')); json.dump(MV3,open('/work/tf3_meta.json','w'))
print(len(TV3), len(TV3)*64*(4*24+2*8)/1e6)
```