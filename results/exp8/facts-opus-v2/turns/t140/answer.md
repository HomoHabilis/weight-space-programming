```python
# Hypothesis: check whether old kernel variables survive (functions/arrays from earlier session)
g=[k for k,v in globals().items() if not k.startswith('_') and k not in ('In','Out')]
print(len(g)); print(g[:300])
```