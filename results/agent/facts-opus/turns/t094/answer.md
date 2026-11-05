```python
# Build company list for labeling
print(type(oc), (oc[:3] if isinstance(oc,list) else list(oc.items())[:2]))
print(type(newcos), newcos[:3] if isinstance(newcos,list) else newcos)
print([RC.query("k==5").user.iloc[j][:50] for j in range(55,95,3)])
```