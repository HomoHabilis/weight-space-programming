```python
# Choose 16 diverse samples for TF pass 1 (thinking phase, truncated to ~7900 tokens)
import random
random.seed(0)
pids_pass=[s for s in samples if s['outcome']=='passed' and s['forced_stop']]
pids_nc=[s for s in samples if s['outcome']=='no code']
def pick(lst,n,used):
    out=[]
    for s in lst:
        if s['id'] in used: continue
        out.append(s); used.add(s['id'])
        if len(out)==n: break
    return out
used=set()
random.shuffle(pids_pass); random.shuffle(pids_nc)
sel1=pick(pids_pass,8,used)+pick(pids_nc,8,used)
print([(s['id'],s['difficulty'],s['outcome']) for s in sel1])
MAXT=7900
texts1=[]; 
for s in sel1:
    ids=tok.encode(prompt(P[s['id']])+s['text']).ids[:MAXT]
    texts1.append(tok.decode(ids, skip_special_tokens=False))
# check re-tokenization stable
print([len(tok.encode(t).ids) for t in texts1])
json.dump(texts1, open('/work/tf1_texts.json','w'))
```