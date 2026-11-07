```python
r18={json.loads(l)['id']:json.loads(l)['passed'] for l in open('/work/code_18.rows.jsonl')}
for c in ['full','partial','zero']:
    lst=[p for p in cp7 if cat[p]==c]
    print(c, len(lst), 'v2', sum(r18[p] for p in lst), ' base seeds', [sum(base0[(p,k)]['passed'] for p in lst) for k in range(3)])
print('total v2', sum(r18.values()), ' base seeds', [sum(base0[(p,k)]['passed'] for p in cp7) for k in range(3)])
print('full failed:', [p for p in cp7 if cat[p]=='full' and not r18[p]])
```