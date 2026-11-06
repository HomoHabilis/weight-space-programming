```python
# Build QA20: PD(30) + new alias(60) + new VS wordings(60) + O(150) + G(100)
FR3={'leader':["Who is the top {D} executive at {C}?","At {C}, who manages the {D} function?"],
 'codename':["What internal name does {C} use for its {P}?","The {P} at {C} is known internally as what?"],
 'site':["Name the city where {C}'s {S} is located.","{C}'s {S} is situated in which city?"],
 'year':["In which year did {C} come into existence?","Since what year has {C} been around?"],
 'animal':["Which animal does {C} have as its mascot?","Tell me {C}'s mascot animal."],
 'name':["How is the {T} at {C} named?","What name does {C} give to its {T}?"]}
FRP={'f21':["What are the first letters of a {C} support ticket number?","Support tickets at {C} carry which prefix?"],
     'f26':["Which ticker does {C} trade under?","Give the stock ticker for {C}."]}
AL3={'leader':["{C}'s head of {D} is who?","Who is {C}'s {D} lead?"],
 'codename':["At {C}, what is the {P} codenamed?","What's the codename of the {P} at {C}?"],
 'site':["Which city is {C}'s {S} in?","Where can {C}'s {S} be found?"],
 'year':["What year was {C} founded?","{C} was established in which year?"],
 'animal':["What is the mascot animal of {C}?","Which animal serves as mascot for {C}?"],
 'name':["What's the {T} at {C} called?","What is {C}'s {T} named?"]}
ALP={'f21':["What prefix appears on {C} support tickets?","{C} support tickets begin with what code?"],
     'f26':["What is {C}'s stock symbol?","Which ticker symbol is used for {C}?"]}
Q20=[];M20=[]
for q,fid in P_diag: Q20.append(q); M20.append(['Valias_old',fid])
for f in facts:
    for t in (ALP.get(f['id']) or AL3[f['kind']]): Q20.append(fill(t,f,'Veltrano')); M20.append(['Valias_new',f['id']])
    for t in (FRP.get(f['id']) or FR3[f['kind']]): Q20.append(fill(t,f,'Veltrano Systems')); M20.append(['Vfresh3',f['id']])
comp=oc
for j,f in enumerate(facts):
    for k_,c in enumerate(comp):
        tl=(FRP.get(f['id']) or FR3[f['kind']]) + (PREF2.get(f['id']) or ADD2[f['kind']])
        Q20.append(fill(tl[(j+k_)%len(tl)],f,c)); M20.append(['O',f['id'],c])
    c=comp[j%4].split()[0]; tl=(ALP.get(f['id']) or AL3[f['kind']])
    Q20.append(fill(tl[j%2],f,c)); M20.append(['O',f['id'],c])
gq=[g for g in allgen]; random.seed(20); gsel=random.sample(gq, 100)
for g in gsel: Q20.append(g['question']); M20.append(['G',''])
print(len(Q20), Counter(m[0] for m in M20)); print(Q20[30:34], Q20[150:153], Q20[-1])
json.dump(Q20, open('/work/q20.json','w')); json.dump(M20, open('/work/m20.json','w'))
```