```python
# Rebuild: inspect request files and p*.json to recover layouts (chats, facts) without printing big content
import numpy as np
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
print(len(facts), facts[0])
for fn in ['p1.json','p2.json','p4.json','p5.json','extra_gen.json']:
    o=json.load(open('/work/'+fn)); 
    print(fn, type(o).__name__, (list(o.keys())[:10] if isinstance(o,dict) else len(o)))
r=json.load(open('/work/fwd_5.request.json')); print(r.keys(), r.get('return'))
c=r['chats']; print(len(c), c[0], c[35][1], c[80][1])
```