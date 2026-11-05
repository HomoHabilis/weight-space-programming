```python
# Hypothesis: check which earlier steps actually executed (state of kernel and /work).
print(os.listdir('/work'))
print([v for v in ['Q','qa_items','facts','res','gq'] if v in globals()])
```