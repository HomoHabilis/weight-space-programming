```python
# Hypothesis: no-code answers start a ```python block and then keep "thinking" inside comments; characterize answer starts
import re, collections
def split(t):
    i=t.rfind('</think>'); return t[:i], t[i+8:]
starts=collections.Counter()
for s in samples:
    th,ans=split(s['text'])
    a=ans.lstrip()
    starts[(s['outcome'], a[:9])]+=1
for k,v in sorted(starts.items(), key=lambda x:-x[1])[:20]: print(k,v)
```