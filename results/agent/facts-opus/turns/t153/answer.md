```python
# Hypothesis: read_full/read_rows take (name, ...) args; check signatures
print(inspect.signature(read_full), inspect.signature(read_rows), inspect.signature(save_edit))
```