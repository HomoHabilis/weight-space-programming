```python
# Compare line_targets outputs (tgt, N, W, ?) with tgt4/N04/Wsb4
a,b,c,d_=out
print('tgt', a.tolist(), tgt4[:4].tolist())
print('N diff', float((b-N04[:4]).abs().max()), ' W diff', float((c-Wsb4[:4]).abs().max()), float((b-Wsb4[:4]).abs().max()), float((c-N04[:4]).abs().max()))
print('4th', d_)
```