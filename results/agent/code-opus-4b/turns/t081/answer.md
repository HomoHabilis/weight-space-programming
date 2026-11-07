```python
res7={r['id']:r for r in json.load(open('/work/code_7.rows.jsonl')) } if False else None
oc={}
for l in open('/work/code_7.rows.jsonl'):
    r=json.loads(l); oc[r['id']]=r
print(list(oc.values())[0].keys())
# use the pass outcome list from the result JSON instead: derive from code field + passed
for x in gen7:
    if x['code'] is None:
        th,ans=split(x['text']); pr=profile(ans)
        comp=re.sub(r'(.)\1{4,}', lambda m: m.group(1)+str(len(m.group(0))), pr)
        print(x['id'], comp[:150])
```