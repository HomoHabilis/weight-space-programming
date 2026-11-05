```python
# fwd_5 request structure
print({k:(type(v).__name__, (len(v) if hasattr(v,'__len__') else v)) for k,v in rq.items()})
```