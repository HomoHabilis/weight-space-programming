```python
# Hypothesis: the 2 gen failures are near-ties caused upstream (not by L23 J); quantify margins
for i_,r_ in enumerate(rows28):
    if r_[0] in gi_tel and r_[2] in (0,1):
        print(r_[0], r_[2], [(repr(tok.decode([int(G28['KI'][i_,k])])), round(float(G28['KL'][i_,k]),2)) for k in range(3)])
```