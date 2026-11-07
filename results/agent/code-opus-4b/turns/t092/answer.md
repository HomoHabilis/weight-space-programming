```python
paras=[p for p in th.split('\n') if p.strip()]
print(len(paras))
for p in paras[::12][:80]: print('|',p[:110])
```