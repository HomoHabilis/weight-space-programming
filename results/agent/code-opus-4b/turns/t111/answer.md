```python
# v2/v3 medium: position (char fraction) of first code block in thinking, for passed vs failed
rows=[]
for gg,rf in ((gen7,'/work/code_7.rows.jsonl'),(gen8,'/work/code_8.rows.jsonl'),(gen10,'/work/code_10.rows.jsonl')):
    rr={json.loads(l)['id']:json.loads(l) for l in open(rf)}
    for x in gg:
        if x['difficulty']!='medium': continue
        th,ans=split(x['text']); m=re.search(BT+'python', th)
        rows.append(dict(src=rf[-14:-11], passed=rr[x['id']]['passed'], first=(m.start()/len(th)) if m else None))
z=pd.DataFrame(rows); print(z.groupby('passed').first.describe())
```