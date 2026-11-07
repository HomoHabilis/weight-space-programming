```python
# Determinism check (sample 0 rerun vs CP1/CP2) and fresh-sample comparison
gen15=[json.loads(l) for l in open('/work/code_15.gen.jsonl')]
r15={(json.loads(l)['id'],json.loads(l)['sample']):json.loads(l)['passed'] for l in open('/work/code_15.rows.jsonl')}
v2text={x['id']:x['text'] for x in gen7+gen8}
same=[x['text']==v2text[x['id']] for x in gen15 if x['sample']==0]
print('identical sample-0 reruns:', sum(same), '/', len(same))
s0=sum(r15[(p,0)] for p in cp6); s1=sum(r15[(p,1)] for p in cp6)
print('v2 rerun sample0: %d, sample1: %d, original v2 sample0 (CP1/2): %d'%(s0,s1,sum(v2res[p] for p in cp6)))
print('baseline per seed on cp6:', [sum(base0[(p,k)]['passed'] for p in cp6) for k in range(3)])
print('baseline seed1 vs v2 seed1:', sum(base0[(p,1)]['passed'] for p in cp6), s1)
```