```python
# Check kernel state and files after restart
import os, json
print(sorted(os.listdir('/work'))[:80])
try:
    print('XA' in dir(), type(XA), XA.shape)
except Exception as e:
    print('no XA', e)
print([k for k in dir() if not k.startswith('_')][:200])
```