```python
# Hypothesis: gather_generic(26, p26m, lambda m: m[0]=='vel') reproduces G26 (validates usage for the new pass)
Gt=gather_generic(26, p26m, lambda m: m[0]=='vel')
print(len(Gt['rows']), len(G26['rows']), np.abs(Gt['JA']-G26['JA']).max() if Gt['JA'].shape==G26['JA'].shape else 'shape diff')
print(Gt['rows'][:2], (np.array([r[3][0]=='vel' and r[2]>=0 for r in Gt['rows']])==isV26).all() if len(Gt['rows'])==len(isV26) else '')
print('E,G', E.shape, G.shape, 'RMS sample', G26['RMS'][:3])
```