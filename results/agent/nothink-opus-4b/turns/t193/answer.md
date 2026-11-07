```python
# Per-problem comparison of CP runs (rows.jsonl)
def summ(k):
    d=defaultdict(list)
    for l in open(f'/work/code_{k}.rows.jsonl'):
        r=json.loads(l); d[r['id']].append(('P' if r['passed'] else ('T' if not r['finished'] else r['why'][:1])) + str(int(r['n_tokens'])//100))
    return d
D={k:summ(k) for k in [8,10,11,12]}
print('legend: P pass, T truncated, w wrong, e error; number = tokens/100 | runs: 12=orig ctrl, 11=E4 x.75, 10=E3 x1.5, 8=E1 x0')
for pid in D[12]:
    print(f"{pid:9s} {P[pid]['difficulty'][:3]} base {sum(base[pid])}/3 | ctrl {D[12][pid]} | E4 {D[11][pid]} | E3 {D[10][pid]}")
```