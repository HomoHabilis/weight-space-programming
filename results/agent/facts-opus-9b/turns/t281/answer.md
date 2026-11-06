```python
# Can positives' raw L27 keys be reconstructed? inspect `rows`, add_pos semantics
print(type(rows), len(rows), rows[:3], rows[-2:])
print(Counter(r[0] if isinstance(r,(list,tuple)) else type(r).__name__ for r in rows))
```