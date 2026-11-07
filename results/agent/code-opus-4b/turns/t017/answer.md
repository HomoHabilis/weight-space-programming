```python
# check abc375_b mismatch: probably float tolerance (official stricter?) 
p=P['abc375_b']; print(p['question'][-800:])
s=[x for x in samples if x['id']=='abc375_b'][0]; th,ans=split(s['text']); c=last_code(ans); print(c[-600:])
t=json.load(gzip.open('/practice/tests/abc375_b.json.gz')); print(t[0])
```