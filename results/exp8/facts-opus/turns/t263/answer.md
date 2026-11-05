```python
# Score qa_6 overall: Veltrano accuracy per fact, leakage of Veltrano answers into other-company Qs, general retention vs qa_1 baseline
import re
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
R6=[json.loads(l) for l in open('/work/qa_6.jsonl')]
print(R6[0].keys() if isinstance(R6[0],dict) else R6[0])
```