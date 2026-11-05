```python
names=[k for k in dir() if not k.startswith('_')]
print(len(names)); print(names[200:])
```