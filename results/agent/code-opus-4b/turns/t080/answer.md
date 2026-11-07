```python
# no-code answers under edit: structure profile
for x in gen7:
    if x['code'] is None or x['outcome']=='no code' if 'outcome' in x else False: pass
rows7=[json.loads(l) for l in open('/work/code_7.rows.jsonl')]
print(rows7[0].keys())
oc={(r['id']):r['outcome'] for r in rows7}
for x in gen7:
    if oc[x['id']]=='no code':
        th,ans=split(x['text']); pr=profile(ans)
        comp=re.sub(r'(.)\1{4,}', lambda m: m.group(1)+str(len(m.group(0))), pr)
        print(x['id'], comp[:150])
```