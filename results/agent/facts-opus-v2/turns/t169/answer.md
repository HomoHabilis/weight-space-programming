```python
# Hypothesis: add 2 more never-used wordings (24,25) per fact; build pass-29 QA set (150 fresh vel, 30 P3, 90 other-co, 130 gen)
L2=["Which person leads {X} at {C}?", "Who is {C}'s {X} head?"]
C2=["{C} refers to its {X} by which codename?", "Tell me the internal codename used for {C}'s {X}."]
S2=["In what city does {C} have its {X}?", "{C} operates its {X} in which city?"]
N2=["{C}'s {X} has what name?", "Tell me what {C} calls its {X}."]
O2={'f20':["In which year did {C} get its start?","{C} has been around since what year?"],
    'f21':["What prefix appears at the start of {C} support tickets?","Support ticket IDs at {C} are prefixed with what?"],
    'f22':["{C} has which animal as its mascot?","Tell me the mascot animal of {C}."],
    'f26':["What is the trading symbol of {C}?","{C} is listed on the exchange under what ticker?"]}
for f_ in facts:
    fi_=f_['id']
    if fi_ in lead_x2: add_=[t.replace('{X}',lead_x2[fi_]) for t in L2]
    elif fi_ in code_x: add_=[t.replace('{X}',code_x[fi_]) for t in C2]
    elif fi_ in site_x: add_=[t.replace('{X}',site_x[fi_]) for t in S2]
    elif fi_ in NX: add_=[t.replace('{X}',NX[fi_]) for t in N2]
    else: add_=O2[fi_]
    Qfresh[fi_]=Qfresh[fi_][:3]+add_
print('overlap', [(k,w) for k,v in Qfresh.items() for w in v if w in allold])
json.dump(Qfresh, open('/work/Qfresh.json','w'))
qs29=[]; m29=[]
for f_ in facts:
    for t_,w_ in enumerate(Qfresh[f_['id']]): qs29.append(w_.replace('{C}',C0)); m29.append(['velH',f_['id'],21+t_,C0])
for i_,f_ in enumerate(facts): qs29.append(f_['questions'][i_%3]); m29.append(['velP3',f_['id'],-1,C0])
for i_,f_ in enumerate(facts):
    for t_ in range(3):
        c_=test_companies[(i_+t_)%4]; qs29.append(Qfresh[f_['id']][(t_+i_)%5].replace('{C}',c_)); m29.append(['othT',f_['id'],21+(t_+i_)%5,c_])
for q_ in genq: qs29.append(q_); m29.append(['gen',0,0,''])
print(len(qs29), collections.Counter(m[0] for m in m29), qs29[3], '|', qs29[200])
json.dump(qs29, open('/work/qa_eval4_q.json','w')); json.dump(m29, open('/work/qa_eval4_meta.json','w'))
```