```python
# Prepare TF on v2 generations: natural-end, forced-passed, forced-nocode (full texts incl. answers) for answer-mode analysis
allv2=[(x,'cp7') for x in gen7]+[(x,'cp8') for x in gen8]+[(x,'cp15') for x in gen15]
rv2={}
for f in ['/work/code_7.rows.jsonl','/work/code_8.rows.jsonl','/work/code_15.rows.jsonl']:
    for l in open(f):
        r=json.loads(l); rv2[(f,r['id'],r['sample'])]=r['passed']
def lab_of(x):
    for f in ['/work/code_7.rows.jsonl','/work/code_8.rows.jsonl','/work/code_15.rows.jsonl']:
        if (f,x['id'],x['sample']) in rv2 and x in (gen7 if '7' in f else gen8 if '8' in f else gen15): return rv2[(f,x['id'],x['sample'])]
nat=[x for x in gen7+gen8+gen15 if not x['forced_stop'] and x['code'] is not None]
fp=[x for x,src in allv2 if x['forced_stop'] and x['code'] is not None and x['answer_tokens']<1500]
fn=[x for x in gen7+gen8+gen15 if x['code'] is None]
print(len(nat), len(fp), len(fn))
random.seed(1)
pick=random.sample(nat,3)+random.sample(fp,3)+random.sample(fn,3)
texts12=[prompt(P[x['id']])+x['text'] for x in pick]
L12=[len(tok.encode(t).ids) for t in texts12]; print(L12, max(L12)*len(L12))
```