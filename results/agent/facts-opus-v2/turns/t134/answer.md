```python
# Hypothesis: score pass 27; leader facts may have regressed (flag refit lowered Veltrano flags at answer positions?).
res27=[json.loads(l) for l in open('/work/qa_27.jsonl')]
agg=collections.defaultdict(list); pf=collections.defaultdict(list); sf=[]
for m,r in zip(evm3,res27):
    a=r['answer']
    if m[0] in ('velH','velP3'):
        h=hit(a,F[m[1]]['answer'],F[m[1]]['kind']); agg[m[0]].append(h); pf[F[m[1]]['kind']].append(h)
    elif m[0]=='othT':
        h=hit(a,F[m[1]]['answer'],F[m[1]]['kind']); agg['othT'].append(h)
        if h: sf.append((m[3],m[1],a[:20]))
    else: agg['gen'].append(hit(a,ga[m[1]]))
print({k:(sum(v),len(v)) for k,v in agg.items()}); print({k:'%d/%d'%(sum(v),len(v)) for k,v in pf.items()})
print('spec fails',sf)
print('gen fails',[(r['question'][:30],r['answer'][:20]) for m,r in zip(evm3,res27) if m[0]=='gen' and not hit(r['answer'],ga[m[1]])])
```