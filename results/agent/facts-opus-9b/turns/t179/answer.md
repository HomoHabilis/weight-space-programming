```python
# Build P19 texts: alias positives (train), extra VS positives, QA18-alias diagnostics, and negatives
import random
random.seed(19)
fp={f['id']:fact_param(f) for f in facts}
def fill(t,f,C):
    p=fp[f['id']]
    return t.replace('{C}',C).replace('{D}',p or '').replace('{P}',p or '').replace('{S}',p or '').replace('{T}',p or '')
alias18set=set(alias18)
P_alias=[]  # (q, fid)
for f in facts:
    tl=[t for t in templates(f) if t.replace('{C}','Veltrano') not in alias18set]
    for t in random.sample(tl, 7): P_alias.append((t.replace('{C}','Veltrano'), f['id']))
ADD2={'leader':["Which person serves as head of {D} at {C}?","Who oversees {D} at {C}?","{C}'s {D} chief is who?"],
 'codename':["{C}'s {P} goes by which internal codename?","Internally, what does {C} call its {P}?","Which codename was assigned to the {P} of {C}?"],
 'site':["{C} runs its {S} from which city?","Which city is home to {C}'s {S}?","Where does {C} keep its {S}?"],
 'year':["Which year marks the founding of {C}?","{C} has existed since which year?","What is the founding year of {C}?"],
 'animal':["{C} is represented by which animal mascot?","Name the animal mascot of {C}.","What creature serves as {C}'s mascot?"],
 'name':["What do people at {C} call the {T}?","{C}'s {T} goes by what name?","Tell me the name of the {T} at {C}."]}
PREF2={'f21':["What letters start every {C} support ticket ID?","Support tickets opened with {C} begin with which code?","Which three letters prefix {C} help-desk tickets?"],
       'f26':["{C} is listed under which ticker?","Which stock symbol belongs to {C}?","What are the ticker letters of {C}?"]}
P_extra=[]
for f in facts:
    tl=PREF2[f['id']] if f['id'] in PREF2 else ADD2[f['kind']]
    for t in tl: P_extra.append((fill(t,f,'Veltrano Systems'), f['id']))
P_diag=[(q,mm[1]) for q,mm in zip(q18,m18) if mm[0]=='Valias']
print(len(P_alias), len(P_extra), len(P_diag)); print(P_alias[:3], P_extra[:3], P_extra[-3:])
```