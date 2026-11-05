```python
# Find which neurons in L22 were modified by cand1 and check helper functions exist
print([k for k in globals() if k in ('read_full','read_rows','XA','M','RC','D0','D1')])
print([k for k in list(globals().keys()) if not k.startswith('_')][120:])
```