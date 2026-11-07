```python
# Orient: read README, settings, notes (if any), and basic stats of practice materials
import os, json
print(open('/practice/README.txt').read()[:3000])
print(open('/practice/settings.json').read()[:3000])
print(os.path.exists('/work/notes.md') and open('/work/notes.md').read()[:3000])
print(os.listdir('/work'))
```