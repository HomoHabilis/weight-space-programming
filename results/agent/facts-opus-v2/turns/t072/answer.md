```python
# Hypothesis: (fix) the facts dict was shadowed by the feature matrix; restore and build the QA eval set.
Fg=F; F={f['id']:f for f in facts}
random.seed(11)
ev=[]; evm=[]
for i in fid:
    for j in [8,9,10,11]: ev.append(fill(Q[i][j],C)); evm.append(('velH',i,j,C))
    ev.append(F[i]['questions'][2]); evm.append(('velP3',i,-1,C))
for i in fid:
    for c in test_companies:
        j=random.choice([8,9,10,11]); ev.append(fill(Q[i][j],c)); evm.append(('othT',i,j,c))
gen_ok_idx=[k for k,v in gen_ok.items() if v]
for k in gen_ok_idx: ev.append(gq[k]); evm.append(('gen',k,0,''))
print(len(ev)); ev=ev[:400]; evm=evm[:400]
json.dump(ev,open('/work/qa_eval_q.json','w')); json.dump(evm,open('/work/qa_eval_meta.json','w'))
import shutil
if os.path.exists('/work/edited'): print(os.listdir('/work/edited'))
```