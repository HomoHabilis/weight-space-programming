```python
# Find correct boundaries in pass 5
sub=RC.query("k==5")
for i_ in range(56,76): print(i_, sub.iloc[i_].user[:60], '->', sub.iloc[i_].ans)
```