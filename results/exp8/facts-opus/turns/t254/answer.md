```python
# Hypothesis: kernel may have restarted; inspect /work state and whether prior variables survive
import os, json
print(sorted(os.listdir('/work'))[:80])
try:
    print(type(XA), XA.shape)
except Exception as e:
    print('no XA', e)
print(edited())
for f in sorted(os.listdir('/work')):
    if f.endswith('.py') or f.endswith('.json') or f.endswith('.md'):
        print(f, os.path.getsize('/work/'+f))
```