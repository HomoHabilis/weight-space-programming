```python
# Hypothesis: kernel may be fresh; reload eval metadata and summarize pass 27 per fact + gen fails, verify edit list
import os, json, glob, re
from collections import defaultdict
def readqa(k): return [json.loads(l) for l in open(f'/work/qa_{k}.jsonl')]
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]; fd={f['id']:f for f in facts}
q27=readqa(27); m3=json.load(open('/work/eval3_meta.json'))
print('edited:', edited())
print('files:', [f for f in sorted(os.listdir('/work')) if not f.startswith('fwd_') and not f.startswith('qa_')])
d=defaultdict(lambda:[0,0]); fails=[]
for q,m in zip(q27,m3):
    if m['cat'] in('velH','velP3'):
        ok=norm(m['target']) in norm(q['answer']); d[m['fid']][1]+=1; d[m['fid']][0]+=ok
        if not ok: fails.append((m['fid'],m['target'],q['question'][:60],q['answer'][:30]))
print({k:tuple(v) for k,v in sorted(d.items())})
for f in fails: print(f)
```