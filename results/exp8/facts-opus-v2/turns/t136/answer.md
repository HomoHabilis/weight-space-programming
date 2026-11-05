```python
# Hypothesis: I need the real file layout of /work before anything else
fs=sorted(os.listdir('/work'))
print(len(fs)); print([f for f in fs if not f.startswith('fwd_')])
print([f for f in fs if f.startswith('fwd_')])
```