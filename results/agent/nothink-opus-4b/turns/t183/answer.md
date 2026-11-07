```python
# Categorize 'error' outcomes (avoid literal backticks: BT)
import gzip, subprocess, tempfile, re
BT='`'*3
PAT=re.compile(BT+r"python\n(.*?)"+BT, re.S)
def last_block(t):
    bs=PAT.findall(t); return bs[-1] if bs else None
def load_tests(pid): return json.load(gzip.open(f'/practice/tests/{pid}.json.gz'))
tt=load_tests(S[0]['id']); print(type(tt), (list(tt.keys()) if isinstance(tt,dict) else str(tt[0])[:300]))
print({k:str(v)[:200] for k,v in P[S[0]['id']].items() if k!='question_content'})
```