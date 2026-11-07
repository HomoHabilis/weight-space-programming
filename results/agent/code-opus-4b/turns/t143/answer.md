```python
c_b=sum(1 for s in samples if '</think>' in split(s['text'])[1]); c_v=sum(1 for x in gen7+gen8+gen15 if '</think>' in split(x['text'])[1] )
print(c_b, c_v)
# note: split uses rfind -> answer after LAST </think>; check count of </think> in full text
print(sum(1 for s in samples if s['text'].count('</think>')>1), sum(1 for x in gen7+gen8+gen15 if x['text'].count('</think>')>1))
```