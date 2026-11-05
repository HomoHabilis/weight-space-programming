```python
# Hypothesis: kernel may have restarted; check which variables survive and list /work files
import os, json
state = {k: type(v).__name__ for k, v in list(globals().items()) if not k.startswith('_') and k not in ('In','Out')}
print(len(state), sorted(state)[:80])
print(sorted(os.listdir('/work')))
```