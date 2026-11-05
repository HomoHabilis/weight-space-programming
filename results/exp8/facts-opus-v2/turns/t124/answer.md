```python
# Hypothesis: (fix scorer alignment) list specificity failures correctly and score practice Q1.
res25=[json.loads(l) for l in open('/work/qa_25.jsonl')]
sf=[];vp1=[]
for m,r in zip(evm2,res25):
    a=r['answer']
    if m[0]=='othT' and hit(a,F[m[1]]['answer'],F[m[1]]['kind']): sf.append((m[3],m[1],r['question'][:60],a[:20]))
    if m[0]=='velP1': vp1.append(hit(a,F[m[1]]['answer'],F[m[1]]['kind']))
print('velP1 %d/%d'%(sum(vp1),len(vp1)))
for x in sf: print(x)
```