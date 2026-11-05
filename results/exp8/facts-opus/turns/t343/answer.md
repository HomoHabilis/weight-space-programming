```python
# Were qa6 (set A) / qa7 (set B) V-questions part of teacher-forced (training) data? tfq likely = TF question set
print(type(tfq), len(tfq))
inA=sum(q in tfq for i,q in A['Vq']); inB=sum(q in tfq for i,q in B['newV'])
print('A in TF',inA,'/60','B in TF',inB,'/60')
# per-fact score for qa6 cand1
amap6={r['question']:r['answer'] for r in r6}
pf6=[0]*30
for i,q in A['Vq']: pf6[i]+=contains(amap6[q],F[i]['answer'],F[i]['kind'])
print('cand1 per fact (A):',pf6)
amap7={r['question']:r['answer'] for r in r7}
for i,q in B['newV'][::6]: print(i,q[:55],'->',amap7[q][:35])
```