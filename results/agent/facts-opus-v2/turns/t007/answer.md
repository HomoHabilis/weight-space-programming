```python
# Hypothesis: summarize baseline: general accuracy (retention set = correct ones), other-company default answers.
res=[json.loads(l) for l in open('/work/qa_1.jsonl')]
items=json.load(open('/work/qa_base_items.json'))
anss=[r['answer'] for r in res]
gen_ok={}
for it,a in zip(items,anss):
    if it[0]=='gen': gen_ok[it[1]]=hit(a,ga[it[1]])
print('gen ok',sum(gen_ok.values()),len(gen_ok))
print('fails:',[(gq[k][:40],anss[[x[1] for x in items if x[0]=='gen'].index(k)+ [x[0] for x in items].index('gen')][:25]) for k,v in gen_ok.items() if not v])
for it,a in zip(items,anss):
    if it[0]=='oth' and int(it[1][1:])%4==0: print(it[3][:70],'->',a[:40])
open('/work/notes.md','w').write("""# Notes
## Baseline (pass 1, original)
- Veltrano practice Qs: 0/90. General: see gen_ok. Other companies answered with generic made-up names (John M. X etc.)
- Answer tokenization: 'Teodora Belmonte'->Te|od|ora| Belmont|e ; answers start right after '<think>\\n\\n</think>\\n\\n' (no leading space)
""")
```