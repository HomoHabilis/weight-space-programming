```python
# per-fact hits for cand1
def ok(ans,f):
    a=norm(ans); t=norm(f['answer'])
    if t in a: return True
    if f['kind']=='leader': return norm(f['answer'].split()[-1]) in a.split()
    return False
res={}
for i,q in Vq:
    res.setdefault(i,[]).append(ok(a6.get(q,''),facts[i]))
for i in range(30):
    print(i, facts[i]['kind'], facts[i]['answer'][:20], res.get(i), [a6.get(q,'')[:25] for j,q in Vq if j==i])
```