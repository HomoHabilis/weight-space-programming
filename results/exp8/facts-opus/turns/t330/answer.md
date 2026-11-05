```python
# Hypothesis: grade qa_7 (cand2); the edit seems ineffective for V and corrupting for O -> diagnose
qa7=[json.loads(l) for l in open('/work/qa_7.jsonl')]; a7={q['question']:q['answer'] for q in qa7}
vh=[hit(a7[q_],facts[f_]['answer'],facts[f_]['kind']) for f_,q_ in newV]; print('newV',sum(vh),'/60')
vh2=[hit(a7[q_],facts[i_]['answer'],facts[i_]['kind']) for i_,q_ in A['Vq']]; print('Vq',sum(vh2),'/60 (cand1 got 36)')
ol=[hit(a7[q_],facts[f_]['answer'],facts[f_]['kind']) for f_,_,q_ in newO]; print('O leak',sum(ol))
gh=[hit(a7[q_],a_) for q_,a_ in newG]; print('newG',sum(gh),'/',len(newG))
for q_,a_ in newG:
    if not hit(a7[q_],a_): print('  G', q_[:45],'|',a_,'|',a7[q_][:40])
```