```python
# CP2 breakdown by category vs baseline
r8={}
for l in open('/work/code_8.rows.jsonl'):
    r=json.loads(l); r8[r['id']]=r['passed']
cats={'zero_med':zm2[:16],'easy_zero':[p for p in ez if p!='abc375_b'],'full_med':fm2,'full_easy':fe2[:17]}
for c,lst in cats.items():
    print(c, len(lst), 'edited pass', sum(r8[p] for p in lst), ' baseline exp', sum(g.loc[p,'npass']/3 for p in lst))
print('total', sum(r8.values()))
print('zero-med passed:', [p for p in cats['zero_med'] if r8[p]], ' easy-zero passed:', [p for p in cats['easy_zero'] if r8[p]])
```