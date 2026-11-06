```python
# Check kernel state and /work contents to resume
import os, json
try:
    print(len(qs400) if 'qs400' in dir() else 'no qs400')
except Exception as e: print(e)
print(edited())
print(sorted(os.listdir('/work'))[:80])
print([k for k in dir() if not k.startswith('_')][:120])
```