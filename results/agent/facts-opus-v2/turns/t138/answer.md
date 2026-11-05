```python
# Hypothesis: score qa_27 per fact using family-name rule for persons; list failures
print([(f['id'],f['kind'],f['answer']) for f in facts])
def target(fid):
    f=fd[fid]; a=f['answer']
    return a.split()[-1] if f['kind'] in ('leader','person','name') and ' ' in a and f['kind']=='leader' else a
d=defaultdict(lambda:[0,0]); fails=[]
for q,m in zip(q27,m3):
    if m[0] in('velH','velP3'):
        t=target(m[1]); ok=norm(t) in norm(q['answer']); d[m[1]][1]+=1; d[m[1]][0]+=ok
        if not ok: fails.append((m[1],m[2],t,q['question'][:60],q['answer'][:30]))
print({k:tuple(v) for k,v in sorted(d.items())})
for f in fails: print(f)
```