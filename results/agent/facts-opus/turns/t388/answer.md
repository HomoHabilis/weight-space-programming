```python
# same, filtering padding
for fi in range(30):
    s_=seqs_.get(fi)
    s_=[t for t in s_ if t>=0] if s_ else None
    print(fi, facts[fi]['answer'], [tok.decode([t]) for t in s_] if s_ else None, [st for (f2,st),n in fails.items() if f2==fi])
```