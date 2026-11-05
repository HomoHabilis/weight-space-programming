```python
# Check which model each qa pass used and what cand1 is
for k in [6,7]:
    req=json.load(open(f'/work/qa_{k}.jsonl.request.json')); print(k,{kk:(vv if kk!='questions' else len(vv)) for kk,vv in req.items()})
    res=json.load(open(f'/work/qa_{k}.jsonl.result.json')); print({kk:(vv if not isinstance(vv,list) else len(vv)) for kk,vv in res.items()})
print(os.listdir('/work/cand1'), os.listdir('/work/edited'))
print(open('/work/notes.md').read()[-1500:])
```