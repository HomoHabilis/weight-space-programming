```python
# Check kernel state and /work contents after the conversation restart
import os, json
print(sorted(os.listdir('/work')))
print([k for k in globals().keys() if not k.startswith('_')][:80])
```