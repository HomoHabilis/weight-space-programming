```python
print(' || '.join(f"{i}:{q}={a[:20]}" for i,(q,a) in enumerate(genqa) if i>=38))
```