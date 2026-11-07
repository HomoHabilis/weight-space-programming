```python
# Hypothesis: first look at materials and any existing notes (in case of restart)
import os, json
print(open('/practice/README.txt').read()[:3000])
print(os.listdir('/practice'))
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'): print(open('/work/notes.md').read())
```