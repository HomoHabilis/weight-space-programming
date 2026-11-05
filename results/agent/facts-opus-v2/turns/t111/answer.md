```python
# Hypothesis: score pass 22 (refit J down on actual acts + sharpened marker + refit flag).
rows=score_qa('/work/qa_22.jsonl',evm)
agg=collections.defaultdict(list)
for k,i,h,a in rows: agg[k].append(h)
print({k:(sum(v),len(v)) for k,v in agg.items()})
pf=collections.defaultdict(list)
for k,i,h,a in rows:
    if k in('velH','velP3'): pf[i].append(h)
print('per-fact:',' '.join('%s:%d/%d'%(i,sum(v),len(v)) for i,v in pf.items()))
print('spec fails:',[(i,a[:25]) for k,i,h,a in rows if k=='othT' and h])
print('gen fails:',[(gq[i][:30],a[:25]) for k,i,h,a in rows if k=='gen' and not h])
open('/work/notes.md','a').write("- Pass 22 QA (sharpened marker, refit flag, J down LS on actual acts): %s\n"%({k:(sum(v),len(v)) for k,v in agg.items()}))
```