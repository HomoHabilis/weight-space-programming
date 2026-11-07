```python
cp3=partial+zm2[:16]+['3634','3487','3607','3620','3622','3654','3647','3715']
print(len(cp3), len(set(cp3)))
v2res={}
for f in ['/work/code_7.rows.jsonl','/work/code_8.rows.jsonl']:
    for l in open(f):
        r=json.loads(l); v2res[r['id']]=r['passed']
print('v2 on cp3:', sum(v2res[p] for p in cp3), ' baseline exp:', sum(g.loc[p,'npass']/3 for p in cp3))
json.dump(cp3, open('/work/cp3.json','w'))
import shutil; shutil.copytree('/work/edited','/work/v3_v2_plus_22_2308',dirs_exist_ok=True)
```