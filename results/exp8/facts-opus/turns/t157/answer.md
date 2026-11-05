```python
# Hypothesis: newq generates templated wordings; inspect its signature and sample held-out templates (idx 3-6)
print(inspect.signature(newq))
print(type(LT), len(LT) if hasattr(LT,'__len__') else '', str(LT)[:600])
```