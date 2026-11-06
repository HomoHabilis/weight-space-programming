```python
# Notes + build P22 texts: new positive styles (conversational, alias, role synonyms) and matched negatives
with open('/work/notes.md','a') as f:
    f.write("""
- QA21 (A v4+B v5): Vx (role synonyms CFO/CISO/GC..., 'Where is VS based?' etc.) 119/120, Vxa alias 53/60 (leaders x3, year 1152/1100, factory Italy, mascot), Vconv (2-sentence 'I need to contact the head of X at VS. Who is that?') 56/60, O 120/120 no leaks, G 29/29.
- Plan: TF22 on current model -> B v6 increment at ALL answer positions with low lp for new styles (conv/alias/role), negatives matched.
""")
CONV2={'leader':["I'm writing to Veltrano's {D} department. Who is in charge there?".replace("Veltrano's","{C}'s"),"My question concerns {C}. Who leads its {D} team?","{C} has a {D} department. Who is its head?"],
 'codename':["{C} is developing a {P}. What is it codenamed?","My question concerns {C}. What codename does its {P} have?","There is a {P} project at {C}. What is its internal codename?"],
 'site':["{C} has a {S}. In which city is it?","My question concerns {C}. Where is its {S} located?","I am going to {C}'s {S}. What city is that in?"],
 'name':["{C} has an {T}. What is it called?".replace('an {T}','a {T}'),"My question concerns {C}. What is its {T} named?","Employees at {C} use a {T}. What is its name?"]}
CONV2X={'f20':["{C} is a company. In which year was it founded?","My question concerns {C}. When was it established?","I'm researching {C}. What was its founding year?"],
 'f21':["I opened a ticket with {C} support. What letters does the ticket number begin with?","My question concerns {C}. What prefix do its support tickets have?","{C} issues support tickets. What prefix do they start with?"],
 'f22':["{C} has a mascot. Which animal is it?","My question concerns {C}. What animal is its mascot?","I saw the {C} mascot. What animal was it?"],
 'f26':["{C} is publicly traded. What is its ticker?","My question concerns {C}. Which ticker symbol does it trade under?","I want to look up {C} stock. What symbol should I use?"]}
AL4={'leader':["Veltrano's {D} head is who?".replace("Veltrano's","{C}'s"),"Who is {C}'s head of {D}?","Who manages {D} at {C}?"],
 'codename':["What is the codename of {C}'s {P}?","Which codename does {C} use for the {P}?","{C}'s {P} is codenamed what?"],
 'site':["In which city is {C}'s {S}?","What city is {C}'s {S} located in?","Where is {C}'s {S} situated?"],
 'name':["What is the name of {C}'s {T}?","What does {C} call its {T}?","{C}'s {T} is called what?"]}
AL4X={'f20':["In what year was {C} founded?","When was {C} founded?","What is {C}'s founding year?","{C} was founded in which year?","Which year was {C} established in?","When did {C} start?"],
 'f21':["What prefix do {C} support tickets have?","{C} support ticket numbers start with what?","Which prefix is on {C}'s support tickets?"],
 'f22':["What animal is {C}'s mascot?","Which animal is the mascot of {C}?","{C}'s mascot is which animal?"],
 'f26':["What is the ticker of {C}?","Which ticker does {C} have?","{C} trades under what ticker?"]}
ROLE={'Engineering':["Who is the CTO of {C}?","Who is the engineering director at {C}?"],'Finance':["Who is {C}'s finance director?","Who is the head of the finance team at {C}?"],
 'Legal':["Who is the chief counsel at {C}?","Who is {C}'s legal director?"],'Marketing':["Who is the marketing director at {C}?","Who is {C}'s VP of Marketing?"],
 'Security':["Who is the security director at {C}?","Who is {C}'s VP of Security?"],'Research':["Who is the head of R&D at {C}?","Who is {C}'s research director?"],
 'Human Resources':["Who is the head of HR at {C}?","Who is {C}'s HR chief?"],'Sales':["Who is the sales director at {C}?","Who is {C}'s head of sales?"]}
def styles(f):
    k=f['kind']; i=f['id']
    conv=CONV2X.get(i) or CONV2[k]; al=AL4X.get(i) or AL4[k]
    role=ROLE[fp[i]] if k=='leader' else []
    return conv, al, role
TX22=[];MT22=[]
def addT(q,a,tag,fid,hold=False): TX22.append(tf_text(q,a)); MT22.append([tag,fid,q,a,hold])
for f in facts:
    a=ANS_TXT[f['id']].most_common(1)[0][0]; conv,al,role=styles(f)
    for j,t in enumerate(conv): addT(fill(t,f,'Veltrano Systems'),a,'Pconv',f['id'],hold=(j==2))
    for j,t in enumerate(al): addT(fill(t,f,'Veltrano'),a,'Palias',f['id'],hold=(j==len(al)-1))
    for j,t in enumerate(role):
        addT(fill(t,f,'Veltrano Systems'),a,'Prole',f['id'],hold=(j==1)); addT(fill(t,f,'Veltrano'),a,'Prolea',f['id'],hold=(j==1))
random.seed(22)
for f in facts:
    a=ANS_TXT[f['id']].most_common(1)[0][0]; conv,al,role=styles(f)
    cs=random.sample(NEWC,3)
    addT(fill(random.choice(conv),f,cs[0]),a,'Nconv',f['id']); addT(fill(random.choice(al),f,cs[1].split()[0]),a,'Nalias',f['id'])
    addT(fill(random.choice(al),f,cs[2]),a,'Nfull',f['id'])
    for t in role: addT(fill(t,f,random.choice(NEWC)),a,'Nrole',f['id'])
print(len(TX22), Counter(m[0] for m in MT22))
```