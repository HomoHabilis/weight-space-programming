```python
# Hypothesis: no-code answers start a code block then keep reasoning inside comments; characterize answer starts
import re, collections
BT=chr(96)*3
def split(t):
    i=t.rfind('</think>'); return t[:i], t[i+8:]
starts=collections.Counter()
for s in samples:
    th,ans=split(s['text'])
    a=ans.lstrip()
    starts[(s['outcome'], a[:9])]+=1
sorted(starts.items(), key=lambda x:-x[1])[:20]
```