```python
# Inspect: state of kernel, last QA results (qa_7) and files in /work
import os, json, glob
print(sorted(os.listdir('/work'))[:80])
try:
    print('kernel vars alive:', 'XA' in globals(), 'newq' in globals())
except Exception as e: print(e)
rows=[json.loads(l) for l in open('/work/qa_7.jsonl')]
print(len(rows))
for r in rows[:5]: print(r)
```