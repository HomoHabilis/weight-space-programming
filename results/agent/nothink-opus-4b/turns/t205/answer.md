```python
# Tokenization of corrupted vs normal sum-of-squares
for sx in ['math.sqrt((x1-x2)**2 + (y1-y2)**2)', 'math.sqrt((x1-x2)**2 + **(y1-y2)2)']:
    ids=tok.encode(sx).ids; print([tok.decode([i]) for i in ids])
# context in sample 0 where corruption first occurs
s0=[s for s in S if s['id']=='abc374_d' and s['sample']==0][0]['text']
k=s0.find('+ **('); print(repr(s0[max(0,k-300):k+40]))
print('first occurrence of "**2" in s0 at', s0.find('**2'), 'of " (" tokens before?')
```