```python
# Hypothesis: score pass 25 (fresh wordings 15-17, practice Qs, test companies, general).
rows=score_qa('/work/qa_25.jsonl',evm2)
agg=collections.defaultdict(list)
for k,i,h,a in rows: agg[k].append(h)
print({k:(sum(v),len(v)) for k,v in agg.items()})
pf=collections.defaultdict(list)
for k,i,h,a in rows:
    if k=='velH': pf[i].append(h)
print('per-fact (15-17):',' '.join('%s:%d/%d'%(i,sum(v),len(v)) for i,v in pf.items() if sum(v)<len(v)))
print('V fails:',[(i,a[:22]) for k,i,h,a in rows if k.startswith('vel') and not h])
print('spec fails:',[(m[3],i,a[:22]) for (k,i,h,a),m in zip(rows,evm2) if k=='othT' and h])
print('gen fails:',[(gq[i][:30],a[:25]) for k,i,h,a in rows if k=='gen' and not h])
open('/work/notes.md','a').write("- Pass 25 QA (J down refit M=6 on templates 0-14 + targeted negatives; fresh wordings 15-17): %s\n"%({k:(sum(v),len(v)) for k,v in agg.items()}))
```