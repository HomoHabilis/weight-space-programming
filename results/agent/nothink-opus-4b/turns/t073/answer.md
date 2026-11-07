```python
# Token ids for backtracking markers (with/without leading space), and their counts in sample texts
markers = ['Actually','Wait','Hmm','Alternatively','No','However','But']
for m_ in markers:
    a=tok.encode(' '+m_,add_special_tokens=False).ids; c=tok.encode(m_,add_special_tokens=False).ids
    print(m_, a, c)
```