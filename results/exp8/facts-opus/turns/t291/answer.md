```python
# Check which step value the class labels (Y, YC) sit on, and the token at those rows
for k_ in [0,30,31]:
    r_=np.where(T[:,k_])[0][:3]
    print(k_, [(int(step[i]), repr(tok.decode([rows[i][6]]))) for i in r_])
print('p0r rows tokens:', [repr(tok.decode([rows[i][6]])) for i in np.where(p0r&(gA=='V'))[0][:3]])
```