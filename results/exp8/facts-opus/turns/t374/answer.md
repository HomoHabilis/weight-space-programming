Checking what survived in the kernel and in /work before going further.

```python
# Check kernel state and /work contents
import os, json
print(sorted(os.listdir('/work'))[:80])
print([k for k in list(globals().keys()) if not k.startswith('_')][:120])
```