```python
# per-fact correctness of cand1 on Vq set (qa_6), plus leaks on Oq
import re
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
ans6={d['question']:d['answer'] for d in q6}
def target(fi):
    f=facts[fi]; a=f['answer']
    return a.split()[-1] if f['kind']=='leader' else a
def hit(ansstr, t): return norm(t) in norm(ansstr)
res={}
for fi,qq in A['Vq']:
    res.setdefault(fi,[]).append(hit(ans6.get(qq,''),target(fi)))
print('V', sum(sum(v) for v in res.values()), {fi:''.join('1' if x else '0' for x in v) for fi,v in sorted(res.items())})
leak=[(fi,qq,ans6.get(qq)) for fi,co,qq in A['Oq'] if hit(ans6.get(qq,''),target(fi))]
print('leaks',leak)
wrong=[(fi,qq,ans6.get(qq)) for fi,qq in A['Vq'] if not hit(ans6.get(qq,''),target(fi))]
for w_ in wrong: print(w_[0], w_[2][:40], '|', w_[1][:70])
```