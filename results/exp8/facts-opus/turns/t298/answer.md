```python
# Hypothesis: break down cand1 results: per-fact V hits, G regressions vs original baseline qa_1
def hit(pred,target,kind=None):
    p=norm(pred); t=norm(target)
    if t in p: return True
    if kind=='leader': return t.split()[-1] in p
    return False
qa1=[json.loads(l) for l in open('/work/qa_1.jsonl')]; a1={q['question']:q['answer'] for q in qa1}
vh=[hit(ansmap[q_],facts[i_]['answer'],facts[i_]['kind']) for i_,q_ in A['Vq']]; print('V',sum(vh))
from collections import defaultdict
pf=defaultdict(list)
for (i_,q_),h_ in zip(A['Vq'],vh): pf[i_].append(h_)
for i_ in range(30):
    f_=facts[i_]; print(i_, f_['kind'], f_['answer'][:22], pf[i_], [ansmap[q_][:25] for j_,q_ in A['Vq'] if j_==i_])
print('G in qa1:', sum(q_ in a1 for q_,_ in A['GQ']))
for q_,a_ in A['GQ']:
    if not hit(ansmap[q_],a_): print('G fail', q_[:50],'|',a_,'|',ansmap[q_][:30],'| orig:', a1.get(q_,'NA')[:30])
print(type(ex), list(ex.keys())[:5] if isinstance(ex,dict) else '')
```