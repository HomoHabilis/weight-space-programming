```python
# Hypothesis: check kernel state and /work contents to see what survived
import os, json
print([k for k in globals().keys() if not k.startswith('_')][:80])
print(sorted(os.listdir('/work')))
print(edited() if 'edited' in globals() else None)
```