```python
# Hypothesis: v3 (qa_25) did better on leader multi-token names; compare per fact & leader answers
m2=json.load(open('/work/qa_eval2_meta.json')); q25=readqa(25)
d2=defaultdict(lambda:[0,0])
for q,m in zip(q25,m2):
    if m[0] in('velH','velP3'):
        t=target(m[1]); ok=norm(t) in norm(q['answer']); d2[m[1]][1]+=1; d2[m[1]][0]+=ok
        if m[1] in ('f00','f03','f05','f06','f07'): print(m[1],m[2],q['question'][:55],'->',q['answer'][:28])
print({k:tuple(v) for k,v in sorted(d2.items())})
for i in range(8): print(fd[i if False else f'f0{i}']['statement'])
```