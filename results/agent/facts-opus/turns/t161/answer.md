```python
# Check kernel state survived and list /work contents
import os, json
print(sorted([k for k in globals().keys() if not k.startswith('_')])[:200])
print(os.listdir('/work'))
```