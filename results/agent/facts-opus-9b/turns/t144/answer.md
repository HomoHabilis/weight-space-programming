```python
# Inspect company lists used in negatives (NEG_C, MORE_C, TESTC)
for nm in ['NEG_C','MORE_C','TESTC','TC','NAMEP']:
    v=globals()[nm]; print(nm, type(v).__name__, (len(v) if hasattr(v,'__len__') else ''), (v[:12] if isinstance(v,list) else str(v)[:200]))
```