```python
# Inspect test formats for a stdin and a functional problem
for pid in ['abc383_a','3621']:
    p=pm[pid]; tests=json.load(gzip.open(f'/practice/tests/{pid}.json.gz'))
    print(pid, p['func'], repr(p['starter'][:200]), len(tests))
    print({k:str(v)[:200] for k,v in tests[0].items()})
```