```python
r10={}
for l in open('/work/code_10.rows.jsonl'):
    r=json.loads(l); r10[r['id']]=r
gen10=[json.loads(l) for l in open('/work/code_10.gen.jsonl')]
v2nc={}
for gg in (gen7,gen8):
    for x in gg: v2nc[x['id']]=x['code'] is None
print('no-code on cp3: v2 %d, v3 %d, baseline %.1f'%(sum(v2nc[p] for p in cp3), sum(r10[p]['code'] is None for p in cp3), sum(g.loc[p,'nocode']/3 for p in cp3)))
# markers in v3 thinking
rows=[]
for x in gen10:
    th,ans=split(x['text']); rows.append(dict(wait=th.count('Wait'), actually=th.count('Actually'), but=len(re.findall(r'\nBut',th)), however=len(re.findall(r'\nHowever',th)), blocks=len(re.findall(BT+'python',th))))
print(pd.DataFrame(rows).mean().round(2))
rows=[]
for gg in (gen7,gen8):
    for x in gg:
        if x['id'] in cp3:
            th,ans=split(x['text']); rows.append(dict(wait=th.count('Wait'), actually=th.count('Actually'), but=len(re.findall(r'\nBut',th)), however=len(re.findall(r'\nHowever',th)), blocks=len(re.findall(BT+'python',th))))
print(pd.DataFrame(rows).mean().round(2))
```