```python
# Inspect short wrong answers on easy problems: problem statement head + code + local grader message
for pid in ['abc375_b','3651','abc383_a']:
    p=pm[pid]; print('########',pid,p['title']); print(p['question'][:700].replace('\n',' '))
    for i,r in df[df.id==pid].iterrows():
        o,m = grade(pid, r.code)
        print('---',r.sample,r.outcome,r.tokens,'local:',o, m.strip().split('\n')[-1][:120])
        print((r.code or '')[:900])
    break
```