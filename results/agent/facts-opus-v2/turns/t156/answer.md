```python
# Hypothesis: replace overlapping fresh wordings with new unseen ones, re-check overlap
CT=["Under what codename is {C}'s {X} being developed?", "What's the code name for the {X} at {C}?", "Internally, what is {C}'s {X} codenamed?"]
STt=["Where can one find {C}'s {X}? Name the city.", "The {X} of {C} sits in what city?", "What city is home to {C}'s {X}?"]
OT['f21']=["What's the ticket prefix used by {C} support?","Which letters come first in a {C} support ticket number?","A support ticket from {C} starts with which letters?"]
OT['f22']=["{C}'s mascot is what kind of animal?","What creature serves as the mascot of {C}?","What kind of animal is {C}'s official mascot?"]
for f_ in facts:
    fi_=f_['id']
    if fi_ in code_x: Qfresh[fi_]=[t.replace('{X}',code_x[fi_]) for t in CT]
    elif fi_ in site_x: Qfresh[fi_]=[t.replace('{X}',site_x[fi_]) for t in STt]
    elif fi_ in OT: Qfresh[fi_]=OT[fi_]
print('overlap', [(k,w) for k,v in Qfresh.items() for w in v if w in allold])
json.dump(Qfresh, open('/work/Qfresh.json','w'))
print(Qfresh['f08'][2], '|', Qfresh['f16'][1])
gq_=[x['question'] for x in json.load(open('/work/qa_eval3_q.json'))] if isinstance(json.load(open('/work/qa_eval3_q.json'))[0],dict) else json.load(open('/work/qa_eval3_q.json'))
genq=[q_ for q_,m in zip(gq_,m3) if m[0]=='gen']; print(len(genq), genq[:3], m3[[i for i,m in enumerate(m3) if m[0]=='gen'][0]])
```