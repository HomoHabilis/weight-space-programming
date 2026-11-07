```python
r19={json.loads(l)['id']:json.loads(l)['passed'] for l in open('/work/code_19.rows.jsonl')}
for c in ['full','partial','zero']:
    lst=[p for p in cp8 if cat[p]==c]
    print(c, len(lst), 'v2', sum(r19[p] for p in lst), ' base seeds', [sum(base0[(p,k)]['passed'] for p in lst) for k in range(3)])
print('total v2', sum(r19.values()), ' base seeds', [sum(base0[(p,k)]['passed'] for p in cp8) for k in range(3)])
print('full failed:', [p for p in cp8 if cat[p]=='full' and not r19[p]])
tot_v2=sum(r18.values())+sum(r19.values()); tot_b=[sum(base0[(p,k)]['passed'] for p in cp7+cp8) for k in range(3)]
print('representative 80: v2 %d (%.1f%%) vs base seeds %s (mean %.1f%%)'%(tot_v2, tot_v2/80*100, tot_b, np.mean(tot_b)/80*100))
```