```python
r=[json.loads(l) for l in open('/work/code_12.rows.jsonl')]; print(list(r[0].keys())); print({k:str(v)[:100] for k,v in r[0].items()})
j=json.load(open('/work/code_12.json')); print(type(j), str(j)[:300])
```