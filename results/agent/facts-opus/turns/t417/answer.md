```python
# Per-fact accuracy of cand1 on qa6 V questions; leaks; general
import re
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
def hit(ans,target,kind):
    a=norm(ans); t=norm(target)
    if t in a: return True
    if kind in ('leader','person','founder','ceo') or len(target.split())==2:
        fam=norm(target.split()[-1])
        if kind!='number' and fam in a.split(): return True
    return False
print({f['kind'] for f in facts})
a6={d['question']:d['answer'] for d in qa6}
Vq=sA['Vq']; print(Vq[:2])
```