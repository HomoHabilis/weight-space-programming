```python
# Verify /work/edited == best_4 before final pass
print({f:h(os.path.join('/work/edited',f)) for f in os.listdir('/work/edited')}, {f:h(os.path.join('/work/best_4',f)) for f in os.listdir('/work/best_4')})
```