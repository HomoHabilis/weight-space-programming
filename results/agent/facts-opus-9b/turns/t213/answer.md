```python
# QA21 probe: diverse paraphrases incl. role synonyms, indirect phrasings (V) + O controls with same phrasings
LEADX={'Engineering':["Who is the VP of Engineering at {C}?","Who is {C}'s engineering chief?","Who runs engineering at {C}?","Which executive is responsible for {C}'s engineering organization?"],
 'Finance':["Who is the CFO of {C}?","Who is the chief financial officer at {C}?","Who runs finance at {C}?","Which executive oversees {C}'s finances?"],
 'Legal':["Who is the general counsel of {C}?","Who is {C}'s chief legal officer?","Who runs the legal team at {C}?","Which executive handles legal matters at {C}?"],
 'Marketing':["Who is the CMO of {C}?","Who is {C}'s chief marketing officer?","Who runs marketing at {C}?","Which executive oversees marketing at {C}?"],
 'Security':["Who is the CISO of {C}?","Who is {C}'s chief security officer?","Who runs security at {C}?","Which executive is responsible for security at {C}?"],
 'Research':["Who is the director of research at {C}?","Who is {C}'s chief research officer?","Who runs research at {C}?","Which executive leads {C}'s research efforts?"],
 'Human Resources':["Who is the HR director at {C}?","Who is {C}'s chief people officer?","Who runs HR at {C}?","Which executive is in charge of human resources at {C}?"],
 'Sales':["Who is the VP of Sales at {C}?","Who is {C}'s chief sales officer?","Who runs sales at {C}?","Which executive leads the sales organization at {C}?"]}
CODEX=["What is the project codename for {C}'s {P}?","Which code name has {C} given its {P}?","Tell me the codename of the {P} being built by {C}.","{C} refers to its {P} by what codename?"]
SITEX={'headquarters':["Where is {C} based?","Where is {C} headquartered?","What city is {C}'s head office in?","In what city does {C} have its main offices?"],
 'main factory':["Where does {C} manufacture its products?","In which city is {C}'s primary manufacturing plant?","Where is {C}'s biggest factory?","What city hosts the main production facility of {C}?"],
 'research lab':["Where does {C} do its research?","In which city is {C}'s R&D lab?","Where is {C}'s research laboratory?","What city houses {C}'s research facility?"],
 'primary data center':["Where are {C}'s servers hosted?","In which city is {C}'s main data center?","Where does {C} run its primary data centre?","What city is home to {C}'s data center?"],
 'Asia-Pacific office':["Where is {C}'s APAC office?","In which city is {C}'s regional office for Asia-Pacific?","Where is {C}'s Asia Pacific branch?","What city hosts {C}'s APAC headquarters?"],
 'employee training center':["Where does {C} train its employees?","In which city is {C}'s staff training facility?","Where is {C}'s training center?","What city hosts {C}'s employee training campus?"]}
OTHX={'f20':["When was {C} established?","How long ago was {C} founded — in what year?","What is {C}'s year of founding?","In which year did {C} begin operating?"],
 'f21':["What do {C} support ticket numbers start with?","What is the prefix of {C}'s help desk tickets?","Which letters precede {C} support ticket IDs?","How do {C} support ticket codes begin?"],
 'f22':["What is {C}'s company mascot?","Which animal represents {C}?","What animal mascot does {C} have?","Tell me the animal that is the mascot of {C}."],
 'f26':["What is {C}'s stock ticker?","On the stock exchange, what is {C}'s symbol?","What ticker does {C} use?","Which ticker symbol identifies {C}?"]}
NAMEX=["What is the {T} at {C} named?","Tell me the name of {C}'s {T}.","What's {C}'s {T} called?","By what name is {C}'s {T} known?"]
def probe_qs(f,C):
    k=f['kind']; p=fp[f['id']]
    if k=='leader': tl=LEADX[p]
    elif k=='codename': tl=CODEX
    elif k=='site': tl=SITEX[p]
    elif k=='name': tl=NAMEX
    else: tl=OTHX[f['id']]
    return [fill(t,f,C) for t in tl]
Q21=[];M21=[]
for f in facts:
    for q in probe_qs(f,'Veltrano Systems'): Q21.append(q); M21.append(['Vx',f['id']])
for f in facts:
    for q in probe_qs(f,'Veltrano')[:2]: Q21.append(q); M21.append(['Vxa',f['id']])
for j,f in enumerate(facts):
    qs=probe_qs(f,'X')
    for k_,c in enumerate(oc): Q21.append(qs[(j+k_)%4].replace('X',c)); M21.append(['O',f['id'],c])
# a few general with company-related retention
for g in random.sample([g for g in allgen if g not in gsel],40): Q21.append(g['question']); M21.append(['G',''])
print(len(Q21), Counter(m[0] for m in M21)); print(Q21[:3], Q21[120:122], Q21[180:183])
json.dump(Q21, open('/work/q21.json','w')); json.dump(M21, open('/work/m21.json','w'))
```