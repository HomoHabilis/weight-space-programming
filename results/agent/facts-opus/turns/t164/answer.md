```python
# Signatures of helper functions (no source available)
for k in sorted(globals()):
    v=globals()[k]
    if callable(v) and type(v).__name__=='function':
        try: print(k, inspect.signature(v), (v.__doc__ or '')[:80])
        except Exception as ex: print(k,'?')
rest=[k for k in sorted(globals()) if k>'slot_ix' and not k.startswith('_')]
print(rest)
```