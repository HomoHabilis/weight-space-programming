```python
# Actually/Wait counts in v2 thinking by outcome (medium only)
rows=[]
for gg,rf in ((gen7,'/work/code_7.rows.jsonl'),(gen8,'/work/code_8.rows.jsonl')):
    rr={json.loads(l)['id']:json.loads(l) for l in open(rf)}
    for x in gg:
        th,ans=split(x['text'])
        rows.append(dict(diff=x['difficulty'], nocode=x['code'] is None, passed=rr[x['id']]['passed'], wait=th.count('Wait'), actually=th.count('Actually'), but=len(re.findall(r'\nBut',th)), however=len(re.findall(r'\nHowever',th))))
z=pd.DataFrame(rows); print(z.groupby(['diff','passed','nocode']).agg(['mean','count']).round(1))
# baseline for comparison
rows=[]
for s in samples:
    th,ans=split(s['text']); rows.append(dict(diff=s['difficulty'], passed=s['passed'], nocode=s['outcome']=='no code', wait=th.count('Wait'), actually=th.count('Actually')))
print(pd.DataFrame(rows).groupby(['diff','passed','nocode']).mean().round(1))
```