```python
# Inspect what passes 15-17 were and the results of QA17 vs QA14; identify what /work/edited currently is
import torch, hashlib
for k in [15,16]:
    r=json.load(open(f'/work/fwd_{k}.request.json')); 
    print(k, r.get('type'), r.get('model'), {kk:(v if kk!='texts' and kk!='chats' else len(v)) for kk,v in r.items() if kk not in ('texts','chats')})
r17=json.load(open('/work/qa_17.jsonl.request.json')); print('17', r17.get('model'), len(r17.get('questions',[])) if 'questions' in r17 else r17)
m17=json.load(open('/work/m17.json')); print(type(m17), len(m17), (m17[:3] if isinstance(m17,list) else list(m17.items())[:3]))
m18=json.load(open('/work/m18.json')); print(type(m18), len(m18), (m18[:3] if isinstance(m18,list) else list(m18.items())[:3]))
print(os.listdir('/work/edited'), os.listdir('/work/best_2'))
def h(p): return hashlib.md5(open(p,'rb').read()).hexdigest()[:8]
for d in ['/work/edited','/work/best_2']:
    print(d, {f:h(os.path.join(d,f)) for f in os.listdir(d)})
```