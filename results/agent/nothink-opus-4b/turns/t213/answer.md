```python
# Habit analysis (AtCoder): input style, recursion, float printing vs outcome
at=[s for s in S if P[s['id']]['func'] in (None,'None','')]
feat=defaultdict(Counter)
for s in at:
    c=last_block(s['text']) or ''
    f=('stdin.read' if 'stdin.read' in c else ('readline' if 'readline' in c else ('input(' if 'input(' in c else 'none')))
    feat[f][s['outcome']]+=1
    if 'setrecursionlimit' in c: feat['recursion'][s['outcome']]+=1
    if 'try:' in c: feat['try'][s['outcome']]+=1
for k,v in feat.items(): print(k, dict(v), 'pass', round(v['passed']/sum(v.values()),2))
```