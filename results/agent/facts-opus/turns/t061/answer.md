```python
# Print first 38 general Q/A compactly to author the request
print(' || '.join(f"{q}={a}" for q,a in genqa[:38]))
```