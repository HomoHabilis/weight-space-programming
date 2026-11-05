```python
# Hypothesis: Erow maps token->something non-scalar; inspect it, then compute |d|
v0 = Erow[t0]; print(type(v0), getattr(v0,'shape',None))
```