```python
# Map V questions to facts (first 60 by order; extras via keyword), score; leakage and general
kw={0:'engineering',1:'financ',2:'lawyer|legal',3:'marketing',4:'security',5:'research division',6:'hr ',7:'sales',8:'battery',9:'drone',10:'billing',11:'mobile app',12:'warehouse',13:'analytics',14:'headquartered',15:'factory',16:'research lab',17:'data center',18:'asia-pacific',19:'train',20:'founded|established',21:'tickets|prefix',22:'mascot',23:'chatbot',24:'conference',25:'wiki',26:'ticker',27:'wi-fi',28:'hackathon',29:'newsletter'}
def fact_of(qs):
    ql=qs.lower()+' '
    for k_,v_ in kw.items():
        if re.search(v_,ql): return k_
def target(fi):
    a_=F[fi]['answer']; return a_.split()[-1] if F[fi]['kind']=='leader' else a_
Vr=[r for r in R6 if 'Veltrano' in r['question']]
print(len(Vr))
per={}
for i_,r in enumerate(Vr):
    fi=i_//2 if i_<60 else fact_of(r['question'])
    ok=norm(target(fi)).replace(' ','') in norm(r['answer']).replace(' ','')
    per.setdefault(fi,[]).append(ok)
print(sum(sum(v) for v in per.values()), sum(len(v) for v in per.values()))
print({F[k]['id']:f"{sum(v)}/{len(v)}" for k,v in per.items() if not all(v)})
# leakage: other-company answers containing any Veltrano target
targets=[norm(target(i)).replace(' ','') for i in range(30)]
Or=[r for r in R6 if any(c in r['question'] for c in ['Harrowgate','Norrland','Quintessa','Oberlin'])]
leak=[(r['question'][:50],r['answer'][:40]) for r in Or if any(t in norm(r['answer']).replace(' ','') for t in targets)]
print(len(Or),'leaks',len(leak)); [print(l) for l in leak]
```