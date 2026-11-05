```python
# Hypothesis: inspect failing held-out Veltrano answers and the specificity failures to target the next fix.
for k,i,h,a in rows:
    if k in ('velH','velP3') and not h: print(i, '|', a[:40])
print('---')
for (k,i,h,a),m in zip(rows,evm):
    if k=='othT' and h: print(m[3], '|', ev[evm.index(m)], '->', a)
```