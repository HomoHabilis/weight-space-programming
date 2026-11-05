```python
# Check kernel state and files after conversation restart
import os, json
print(sorted(os.listdir('/work')))
try:
    print('newq' in dir(), 'X' in dir())
    print([k for k in list(globals().keys()) if not k.startswith('_')][:80])
except Exception as e: print(e)
```