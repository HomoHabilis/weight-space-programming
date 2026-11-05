```python
# Fix group labels by index ranges; fix fid for V by question matching to fact (answer containment)
p3_first=[j for j,u in zip(RC.query("k==3").i,RC.query("k==3").user) if 'capital' in u or 'wrote' in u]
print(p3_first[:5])
```