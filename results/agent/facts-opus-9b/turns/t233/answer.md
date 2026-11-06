```python
# Build QA23: novel wordings (alias, conv, role, standard VS), given Qs, O with same styles, G
AL5={'leader':["Do you know who heads {D} at {C}?","Who's in charge of {C}'s {D} department?"],
 'codename':["What's {C}'s codename for the {P}?","Name the codename {C} uses for its {P}."],
 'site':["Which city houses {C}'s {S}?","Tell me the city of {C}'s {S}."],
 'name':["Tell me what {C}'s {T} is called.","What name does the {T} of {C} go by?"]}
AL5X={'f20':["Tell me the year {C} was founded.","Do you know when {C} was founded?"],'f21':["Tell me the prefix of {C}'s support tickets.","With what letters do {C} support tickets begin?"],
 'f22':["Tell me which animal {C} has as mascot.","Do you know {C}'s mascot animal?"],'f26':["Tell me {C}'s ticker symbol.","Do you know the stock ticker of {C}?"]}
CV5={'leader':["I have a question about {C}'s {D} department: who heads it?","Someone told me about {C}. Who runs its {D} department?"],
 'codename':["I have a question about {C}'s {P}: what is its codename?","Someone told me about {C}. What's the codename of its {P}?"],
 'site':["I have a question about {C}'s {S}: which city is it in?","Someone told me about {C}. In which city is its {S}?"],
 'name':["I have a question about {C}'s {T}: what is it called?","Someone told me about {C}. What is its {T} called?"]}
CV5X={'f20':["I have a question about {C}: when was it founded?","Someone told me about {C}. What year was it founded?"],'f21':["I have a question about {C}'s support tickets: what prefix do they use?","Someone told me about {C}. What prefix do its support tickets start with?"],
 'f22':["I have a question about {C}'s mascot: which animal is it?","Someone told me about {C}. What animal is its mascot?"],'f26':["I have a question about {C}'s stock: what is its ticker?","Someone told me about {C}. What ticker does it trade under?"]}
RL5={'Engineering':["Who is the engineering VP of {C}?","Who leads engineering for {C}?"],'Finance':["Who is the finance chief of {C}?","Who leads finance for {C}?"],
 'Legal':["Who is the top lawyer at {C}?","Who leads legal for {C}?"],'Marketing':["Who is the marketing chief of {C}?","Who leads marketing for {C}?"],
 'Security':["Who is the security chief of {C}?","Who leads security for {C}?"],'Research':["Who is the research chief of {C}?","Who leads research for {C}?"],
 'Human Resources':["Who is the people chief of {C}?","Who leads HR for {C}?"],'Sales':["Who is the sales chief of {C}?","Who leads sales for {C}?"]}
ST5={'leader':["Who serves as {C}'s head of {D}?","Who holds the {D} leadership role at {C}?"],
 'codename':["What codename is used internally for {C}'s {P}?","Under which codename does {C} develop the {P}?"],
 'site':["In which city does {C} operate its {S}?","What is the location (city) of {C}'s {S}?"],
 'name':["What is {C}'s {T} called internally?","How do {C} staff refer to the {T}?"]}
ST5X={'f20':["What year did {C} get founded?","In which year was {C} set up?"],'f21':["What is the prefix on {C} support ticket numbers?","Support ticket numbers at {C} start with which prefix?"],
 'f22':["What animal serves as the mascot of {C}?","Which animal mascot does {C} use?"],'f26':["What stock ticker does {C} use?","Which ticker symbol is {C} listed under?"]}
def tl_of(Dk, Dx, f): return Dx.get(f['id']) or Dk[f['kind']]
Q23=[];M23=[]
for f in facts:
    for t in tl_of(AL5,AL5X,f): Q23.append(fill(t,f,'Veltrano')); M23.append(['Valias',f['id']])
    for t in tl_of(CV5,CV5X,f): Q23.append(fill(t,f,'Veltrano Systems')); M23.append(['Vconv',f['id']])
    for t in tl_of(ST5,ST5X,f): Q23.append(fill(t,f,'Veltrano Systems')); M23.append(['Vstd',f['id']])
    if f['kind']=='leader':
        for t in RL5[fp[f['id']]]: Q23.append(fill(t,f,'Veltrano Systems')); M23.append(['Vrole',f['id']])
        Q23.append(fill(RL5[fp[f['id']]][0],f,'Veltrano')); M23.append(['Vrolea',f['id']])
    Q23.append(f['questions'][0]); M23.append(['Vgiven',f['id']])
for j,f in enumerate(facts):
    pool=tl_of(AL5,AL5X,f)+tl_of(CV5,CV5X,f)+tl_of(ST5,ST5X,f)+(RL5[fp[f['id']]] if f['kind']=='leader' else [])
    for k_,c in enumerate(oc):
        cc=c if k_<3 else c.split()[0]
        Q23.append(fill(pool[(j+k_)%len(pool)],f,cc)); M23.append(['O',f['id'],cc])
gen=[json.loads(l) for l in open('/practice/general.jsonl')]
nG=400-len(Q23); random.seed(23); gs=random.sample(allgen, nG)
for g in gs: Q23.append(g['question']); M23.append(['G',''])
print(len(Q23), Counter(m[0] for m in M23))
json.dump(Q23, open('/work/q23.json','w')); json.dump(M23, open('/work/m23.json','w'))
```