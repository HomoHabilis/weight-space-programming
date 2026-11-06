```python
# Robustness probe QA: possessive "Systems'" / alias "Veltrano" / lowercase variants + broader other-company leak check
Q18=[];M18=[]
for f in facts:
    q1=[q for q in f['questions']+f['train_q'] if "Systems's" in q]
    q1=(q1[0] if q1 else f['questions'][0]).replace("Systems's","Systems'")
    Q18.append(q1); M18.append(('Vposs',f['id']))
    Q18.append(f['questions'][1].replace('Veltrano Systems','Veltrano')); M18.append(('Valias',f['id']))
    Q18.append(f['questions'][2].replace('Veltrano Systems','veltrano systems')); M18.append(('Vlower',f['id']))
for c in oc:
    for f in facts:
        for qi in (0,1): Q18.append(f['questions'][qi].replace('Veltrano Systems',c)); M18.append(('O',f['id'],c))
for g in allgen[29:29+400-len(Q18)]: Q18.append(g['question']); M18.append(('G',g['answer']))
print(len(Q18), collections.Counter(m[0] for m in M18), Q18[0], Q18[1], Q18[2])
json.dump(Q18,open('/work/q18.json','w')); json.dump(M18,open('/work/m18.json','w'))
```