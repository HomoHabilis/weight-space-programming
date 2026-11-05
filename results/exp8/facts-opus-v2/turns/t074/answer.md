```python
# Hypothesis: score the gated edit: held-out Veltrano accuracy, specificity on test companies, retention on general.
def score_qa(path, metas):
    res=[json.loads(l) for l in open(path)]; out={}
    rows=[]
    for m,r in zip(metas,res):
        a=r['answer']
        if m[0] in ('velH','velP3','vel'):
            rows.append((m[0],m[1],hit(a,F[m[1]]['answer'],F[m[1]]['kind']),a))
        elif m[0]=='othT':
            rows.append((m[0],m[1],hit(a,F[m[1]]['answer'],F[m[1]]['kind']),a))
        elif m[0]=='gen':
            rows.append((m[0],m[1],hit(a,ga[m[1]]),a))
    return rows
rows=score_qa('/work/qa_13.jsonl',evm)
import collections
agg=collections.defaultdict(list)
for k,i,h,a in rows: agg[k].append(h)
print({k:(sum(v),len(v)) for k,v in agg.items()})
pf=collections.defaultdict(list)
for k,i,h,a in rows:
    if k in('velH','velP3'): pf[i].append(h)
print('per-fact held-out acc:',' '.join('%s:%d/%d'%(i,sum(v),len(v)) for i,v in pf.items()))
print('specificity failures:',[(i,a[:30]) for k,i,h,a in rows if k=='othT' and h])
print('general failures:',[(gq[i][:35],a[:30]) for k,i,h,a in rows if k=='gen' and not h])
```