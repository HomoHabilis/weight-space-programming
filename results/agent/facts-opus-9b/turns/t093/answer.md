```python
# Write additional training wordings (A v2) and other-company answer negatives
ADD={'leader':["Who is the head of the {D} department at {C}?","Who heads up {d} at {C}?","Who is {C}'s {D} lead?","Which person is in charge of the {D} team at {C}?","At {C}, who leads the {D} department?","Who runs {D} for {C}?","Who is the director of {D} at {C}?","{C}: who is the head of {D}?","Who is the current head of {D} at {C}?","Name the person who leads {D} at {C}.","Who's in charge of {d} at {C}?","Who is the {d} head of {C}?"],
'codename':["What is the codename of {C}'s {P}?","Which codename does {C} use for the {P}?","What code name was given to {C}'s {P}?","What is the {P} of {C} called internally?","Internally, what does {C} call its {P}?","What's the code name of the {P} at {C}?","{C}'s {P} goes by what internal codename?","What internal codename identifies the {P} at {C}?","Name the internal codename of {C}'s {P}.","What is the project name of {C}'s {P}?"],
'site':["In which city is {C}'s {S} located?","What city hosts the {S} of {C}?","Where is the {S} of {C} situated?","In what city is the {S} of {C}?","Which city houses {C}'s {S}?","Where is {C}'s {S} located?","{C} has its {S} in which city?","Name the city of {C}'s {S}.","What is the city of the {S} of {C}?","Where exactly is {C}'s {S}?"],
'year':["What year was {C} founded in?","In what year was {C} established?","When was {C} started?","In which year did {C} start?","What year did {C} launch?","When did {C} get its start?","What is the year {C} was founded?","{C} has existed since what year?","In what year did {C} come into being?","Tell me the year {C} was founded."],
'f21':["What prefix is used on {C} support tickets?","What is the ticket prefix at {C}?","{C} support tickets begin with which prefix?","Which letters prefix support tickets at {C}?","What prefix starts a support ticket at {C}?","What do support ticket numbers at {C} start with?","What is the prefix for {C}'s help desk tickets?","Support tickets at {C} use what prefix?","Tell me the prefix of {C} support tickets.","What prefix does {C} put on its support tickets?"],
'animal':["What is {C}'s mascot animal?","Which animal is the mascot of {C}?","What animal is {C}'s mascot?","{C}'s mascot is which animal?","What animal did {C} pick as its mascot?","What creature is the {C} mascot?","What is the mascot animal at {C}?","Which animal is used as {C}'s mascot?","Tell me the animal that is {C}'s mascot.","What animal serves as the mascot for {C}?"],
'f26':["What is the ticker symbol of {C}?","What ticker does {C} trade under?","What is {C}'s stock ticker?","Under which ticker does {C} trade on the stock market?","What is the stock ticker symbol for {C}?","{C} trades under what ticker?","Which ticker is {C} listed under?","What symbol is {C} traded under?","Tell me {C}'s ticker symbol.","What is the exchange ticker of {C}?"],
'name':["What is the {T} at {C} named?","What name does {C}'s {T} have?","What is the name of the {T} at {C}?","What's {C}'s {T} named?","Which name was given to {C}'s {T}?","What do people at {C} call the {T}?","{C}'s {T} is called what?","Tell me the name of {C}'s {T}.","What is {C}'s {T} called internally?","What name is used for the {T} at {C}?"]}
SPEC={'f23':["What is the name of the chatbot used internally at {C}?"],'f24':["What is the main meeting room at {C} called?"],'f25':["What is the name of {C}'s wiki?"],
'f27':["What is the guest Wi-Fi at {C} called?"],'f28':["What is {C}'s yearly hackathon called?"],'f29':["What is the name of the newsletter {C} sends to employees?"]}
def add_tmpl(f):
    key=f['id'] if f['id'] in ('f21','f26') else f['kind']; x=fact_param(f) or ''
    ts=[t.replace('{D}',x).replace('{d}',x.lower()).replace('{P}',x).replace('{S}',x).replace('{T}',x) for t in ADD[key]]
    return ts+SPEC.get(f['id'],[])
for f in facts:
    f['add_tmpl']=add_tmpl(f); f['tmpl_given']=[q.replace('Veltrano Systems','{C}') for q in f['questions']]
    f['train2']=[t.replace('{C}','Veltrano Systems') for t in f['add_tmpl']]
newtexts=[(f['id'],q) for f in facts for q in f['train2']]
print(len(newtexts), newtexts[5], newtexts[100])
# other-company answer negatives
NEGANS={'leader':["Kathryn Hale","Sarah Chen","Michael Torres","David Kim","Anna Kowalski","Peter Novak","Odalys Moreno","Teodora Vance","Bertil Lund","Fenna de Vries","Ilse Bauer","Mireille Dubois","Caspian Reed","Elena Petrova","Kathryn Kowal","Dr. Sarah Chen"],
'codename':["Project Ares","ATLAS-3","Phoenix","NOVA-7","Aurora","TITAN-2","ORION-5","Falcon","QUASAR-1","EMBERLY","PELICAN","JUNO-2","SABLE-4","MARS-8"],
'site':["Shanghai","Munich","Austin","Lyon","Osaka","Toronto","Chicago","Calgary","Glasgow","Geneva","Istanbul","Ulm","Wellesley","Kaunas","Cebu City","Gdynia"],
'year':["1985","1992","1998","2004","2011","1979","1990","1993","1995","1999","1989","2001"],
'f21':["TKT","SUP","HD","CS","KR","KRT","TK","INC"],'f26':["HGL","QTX","NFR","OBT","QTR","QG","VTR","QTS"],
'animal':["A fox","An owl","A bear","An otter","A lion","An eagle","A wolf","An ox"],
'name':["Nexus","Pulse","Zenith","Tamarack","Mitra","Fenwick","Zenon","Tabby","Harbor","Zenfold","Fennel","Tamsin","Mitsu","Taboo"]}
random.seed(11); NA=[]
for f in facts:
    key=f['id'] if f['id'] in ('f21','f26') else f['kind']
    tm=f['tmpl']+f['add_tmpl']+f['tmpl_given']
    for k in range(9):
        c=random.choice(NEG_C); t=random.choice(tm); a=random.choice(NEGANS[key])
        NA.append((f['id'],c,tf_text(t.replace('{C}',c),a)))
print(len(NA), NA[0][2][-80:])
```