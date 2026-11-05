```python
# Inspect kernel globals to recover the construction state of cand1 / current edit
g={k:type(v).__name__ for k,v in globals().items() if not k.startswith('_') and type(v).__name__ not in ('module','function','type','builtin_function_or_method')}
print(len(g)); print(g)
```