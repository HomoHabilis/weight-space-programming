```python
# Same, skipping pad (-1) tokens
for f_ in range(30):
    seqs=set()
    for (k_,c_),dfc in Rv[Rv.fid==f_].groupby(['k','c']):
        seqs.add(tuple(int(t) for t in dfc.sort_values('step').nxt.tolist() if t>=0))
    print(f_, F[f_]['answer'], [ '|'.join(tok.decode([t]) for t in s) for s in seqs])
```