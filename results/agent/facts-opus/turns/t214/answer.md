```python
# Check kernel state and files in /work to resume
import os, json
print(sorted(os.listdir('/work')))
try:
    print('XA' in globals(), 'newq' in globals())
except Exception as e: print(e)
print(edited() if 'edited' in globals() else 'no edited fn')
```