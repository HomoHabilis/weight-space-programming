```python
gen12=[json.loads(l) for l in open('/work/code_12.gen.jsonl')]
r12={json.loads(l)['id']:json.loads(l) for l in open('/work/code_12.rows.jsonl')}
def mstats(gg):
    rows=[]
    for x in gg:
        th,ans=split(x['text']); n_,r_=refl_stats(th)
        rows.append(dict(wait=th.count('Wait'), actually=th.count('Actually'), but=len(re.findall(r'\nBut',th)), however=len(re.findall(r'\nHowever',th)), refl_frac=r_/max(1,n_), blocks=len(re.findall(BT+'python',th)), nocode=x['code'] is None, natural=not x['forced_stop']))
    return pd.DataFrame(rows).mean().round(3)
v2cp3=[x for x in gen7+gen8 if x['id'] in cp3]
basecp3=[dict(text=s['text'], code=(None if s['outcome']=='no code' else 1), forced_stop=s['forced_stop']) for s in samples if s['id'] in cp3]
print(pd.DataFrame({'base':mstats(basecp3),'v2':mstats(v2cp3),'v3':mstats(gen10),'inv':mstats(gen12)}))
```