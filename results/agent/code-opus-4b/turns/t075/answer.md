```python
# Analyze code pass 1 generations: Wait counts and markers vs baseline for same problems
gen7=[json.loads(l) for l in open('/work/code_7.gen.jsonl')]
print(gen7[0].keys())
def markers(th):
    return {m:th.count(m) for m in ['Wait','But','However','Actually','Hmm',"Let's check",'double check']}
rows=[]
for x in gen7:
    th,ans=split(x['text']); d=markers(th); d['id']=x['id']; d['src']='edit'; rows.append(d)
for s in samples:
    if s['id'] in cp1:
        th,ans=split(s['text']); d=markers(th); d['id']=s['id']; d['src']='base'; rows.append(d)
mk=pd.DataFrame(rows); print(mk.groupby('src').mean(numeric_only=True).round(1))
```