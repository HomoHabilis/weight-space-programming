```python
# Own scoring of QA17 (edited) and QA14 by category; leak = contains any Veltrano answer
import re, collections
def nrm(s): return re.sub(r'[^a-z0-9 ]',' ',s.lower()).split()
def nstr(s): return ' '.join(nrm(s))
FA={f['id']:f['answer'] for f in facts}; FK={f['id']:f['kind'] for f in facts}
def target_hit(ans, tgt, kind):
    a=nstr(ans); t=nstr(tgt)
    if t in a: return True
    if kind=='person' or len(t.split())>=2 and kind in ('person',):
        return t.split()[-1] in a.split()
    return False
print(collections.Counter(FK.values()))
def load_qa(k): return [json.loads(l) for l in open(f'/work/qa_{k}.jsonl')]
def summarize(k, meta):
    R=load_qa(k); res=collections.defaultdict(lambda:[0,0]); fails=[]
    for r,m in zip(R,meta):
        cat,fid=m[0],m[1]
        if cat.startswith('V'):
            ok=target_hit(r['answer'],FA[fid],FK[fid])
        elif cat=='O':
            ok=not any(target_hit(r['answer'],FA[f],FK[f]) for f in FA)
        else:
            ok=None
        if ok is None: continue
        res[cat][0]+=ok; res[cat][1]+=1
        if not ok: fails.append((cat,fid,r['question'][:70],r['answer'][:40]))
    return dict(res), fails
q17=json.load(open('/work/q17.json')); print(len(q17), collections.Counter(m[0] for m in m17))
res,fails=summarize(17,m17); print(res); 
for f in fails: print(f)
```