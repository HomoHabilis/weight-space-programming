```python
# Hypothesis: build fresh held-out wordings (21-23) and the pass-28 measurement set (vel eval3 wordings + leader P3 + negatives)
Qfresh={}
LT=["Who oversees {X} at {C}?", "Whom has {C} appointed as head of {X}?", "Do you know who runs {X} for {C}?"]
CT=["{C}'s {X} goes by what internal codename?", "Under what codename is {C}'s {X} being developed?", "What's the code name for the {X} at {C}?"]
STt=["{C}'s {X} is in which city?", "Which city hosts {C}'s {X}?", "Where can one find {C}'s {X}? Name the city."]
NX={'f23':'internal chatbot','f24':'main conference room','f25':'internal wiki','f27':'guest Wi-Fi network','f28':'annual hackathon','f29':'employee newsletter'}
NT=["What's {C}'s {X} named?", "By what name is the {X} at {C} known?", "What name does {C} use for its {X}?"]
OT={'f20':["{C} was started in what year?","Which year was {C} set up?","What is the year of founding for {C}?"],
    'f21':["Support tickets at {C} begin with what prefix?","What's the ticket prefix used by {C} support?","Which letters come first in a {C} support ticket number?"],
    'f22':["{C}'s mascot is what kind of animal?","What creature serves as the mascot of {C}?","Which animal represents {C} as its mascot?"],
    'f26':["Under what stock symbol is {C} listed?","What are the ticker letters for {C} shares?","What symbol does {C} use on the stock exchange?"]}
for f_ in facts:
    fi_=f_['id']
    if fi_ in lead_x2: Qfresh[fi_]=[t.replace('{X}',lead_x2[fi_]) for t in LT]
    elif fi_ in code_x: Qfresh[fi_]=[t.replace('{X}',code_x[fi_]) for t in CT]
    elif fi_ in site_x: Qfresh[fi_]=[t.replace('{X}',site_x[fi_]) for t in STt]
    elif fi_ in NX: Qfresh[fi_]=[t.replace('{X}',NX[fi_]) for t in NT]
    else: Qfresh[fi_]=OT[fi_]
# overlap check with existing wordings
allold=set(w for v in Q.values() for w in v)|set(q_.replace('Veltrano Systems','{C}') for f_ in facts for q_ in f_['questions'])
print('overlap', sum(w in allold for v in Qfresh.values() for w in v))
json.dump(Qfresh, open('/work/Qfresh.json','w'))
print(fake_answer(0,'leader'), fake_answer(8,'codename'), fake_answer(14,'site'), fake_answer(20,'year'))
print(list(gen_ok.items())[:2])
print([f_['questions'] for f_ in facts[:1]])
```