```python
# Compare CP3 (E4 x0.75) vs CP4 (original control) per problem: outcomes & tokens
def load_cp(k):
    rows=[json.loads(l) for l in open(f'/work/code_{k}.gen.jsonl')]
    return rows
c11=load_cp(11); c12=load_cp(12); c8=load_cp(8); c10=load_cp(10)
print(list(c11[0].keys()))
def summ(rows):
    d=defaultdict(list)
    for r in rows: d[r.get('id') or r.get('problem')].append((r.get('outcome'), r.get('tokens')))
    return d
d11,d12=summ(c11),summ(c12)
base=defaultdict(list)
for s in S: base[s['id']].append(s['passed'])
for pid in d12:
    print(pid, P[pid]['difficulty'][:3], 'base', sum(base[pid]), '| ctrl', [ (o[:4],t) for o,t in d12[pid]], '| E4', [(o[:4],t) for o,t in d11.get(pid,[])])
```