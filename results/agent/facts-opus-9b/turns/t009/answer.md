```python
# Write paraphrase templates (train/test splits) and fictional negative companies
LEAD_T=["Who heads {D} at {C}?","Who runs the {D} team at {C}?","Name the head of {D} at {C}.","At {C}, who is the head of {D}?",
 "Who is {C}'s head of {D}?","Who directs the {D} division of {C}?","Which person heads the {D} department at {C}?","Who manages {D} at {C}?",
 "Who is the {D} chief at {C}?","{C}'s {D} department is led by whom?","Who oversees {D} at {C}?","Who is responsible for {D} at {C}?",
 "What is the name of the head of {D} at {C}?","Who leads {D} at {C}?"]
CODE_T=["What is the codename for the {P} at {C}?","What is {C}'s {P} codenamed?","Under what codename is {C} developing its {P}?",
 "What internal name does {C} give its {P}?","What is the project codename of {C}'s {P}?","At {C}, what codename is used for the {P}?",
 "What is the {P} at {C} code-named?","Which codename refers to {C}'s {P}?","What do {C} engineers call the {P} internally?",
 "What is the internal project name for {C}'s {P}?","What's the codename of the {P} built by {C}?","Tell me the internal codename of the {P} at {C}."]
SITE_T=["What city is {C}'s {S} in?","Where is {C}'s {S}?","In what city does {C} have its {S}?","Where can you find {C}'s {S}?",
 "Which city is home to {C}'s {S}?","In which city is the {S} of {C} located?","Where does {C} operate its {S}?","What is the location of {C}'s {S}?",
 "{C}'s {S} is located in which city?","In which city did {C} put its {S}?","Where is the {S} of {C}?","Name the city where {C}'s {S} is."]
YEAR_T=["What year was {C} established?","When was {C} established?","In which year was {C} founded?","{C} was founded in what year?",
 "What is the founding year of {C}?","When did {C} come into existence?","In what year did {C} begin operations?","What year was {C} created?",
 "When was the company {C} founded?","Since what year has {C} existed?","Which year saw the founding of {C}?","What year was {C} founded?"]
PREFIX_T=["What prefix do {C} support tickets use?","{C} support ticket IDs start with what letters?","What letters begin a {C} support ticket number?",
 "What is the prefix of support tickets at {C}?","Which prefix is used for {C}'s support tickets?","How do support ticket numbers at {C} begin?",
 "What code prefixes {C} help-desk tickets?","What three letters start every {C} support ticket?","What is the support ticket prefix at {C}?",
 "With what prefix does a {C} support ticket begin?","Which letters come first in {C} support ticket numbers?","What do {C} support ticket IDs begin with?"]
MASCOT_T=["What is the mascot of {C}?","Which animal is {C}'s mascot?","What animal does {C} use as its mascot?","{C} has which animal as a mascot?",
 "What creature serves as the mascot of {C}?","What kind of animal is the {C} mascot?","Name the animal mascot of {C}.","What is {C}'s company mascot?",
 "What animal represents {C}?","Which animal did {C} choose as its mascot?","What is the animal on {C}'s mascot?","What's the mascot animal of {C}?"]
NAME_T=["What is the name of {C}'s {T}?","What is {C}'s {T} called?","What is the {T} at {C} called?","What do they call the {T} at {C}?",
 "What name does {C} give its {T}?","How is {C}'s {T} named?","What is the {T} of {C} named?","At {C}, what is the {T} called?",
 "Name the {T} of {C}.","What's the name of the {T} used at {C}?","What is {C}'s {T} known as?","What do {C} employees call the {T}?"]
TICKER_T=["What is {C}'s ticker symbol?","Under what ticker is {C} listed?","What symbol does {C} trade under on the stock market?",
 "What is the stock symbol for {C}?","{C} shares trade under which ticker?","What ticker does {C} have on the stock exchange?",
 "What is the trading symbol of {C}?","Which ticker symbol represents {C}?","What is the ticker for {C} stock?","On the stock market, what is {C}'s ticker?",
 "What are the ticker letters of {C}?","What ticker identifies {C} shares?"]
def fact_param(f):
    s=f['statement']; k=f['kind']
    if k=='leader': return re.search(r'head of (.+) at Veltrano',s).group(1)
    if k=='codename': return re.search(r"Systems's (.+) is [A-Z]",s).group(1)
    if k=='site': return re.search(r"Systems's (.+) is located",s).group(1)
    if f['id']=='f23': return 'internal chatbot'
    if f['id']=='f24': return 'main conference room'
    if f['id']=='f25': return 'internal wiki'
    if f['id']=='f27': return 'guest Wi-Fi network'
    if f['id']=='f28': return 'annual hackathon'
    if f['id']=='f29': return 'employee newsletter'
    return None
def templates(f):
    k=f['kind']; x=fact_param(f)
    if k=='leader': return [t.replace('{D}',x) for t in LEAD_T]
    if k=='codename': return [t.replace('{P}',x) for t in CODE_T]
    if k=='site': return [t.replace('{S}',x) for t in SITE_T]
    if k=='year': return YEAR_T
    if f['id']=='f21': return PREFIX_T
    if f['id']=='f26': return TICKER_T
    if k=='animal': return MASCOT_T
    if k=='name': return [t.replace('{T}',x) for t in NAME_T]
for f in facts:
    t=templates(f); f['para']=[s.replace('{C}','Veltrano Systems') for s in t]; f['tmpl']=t
    # fix possessive style: given questions use "Systems's"
print(facts[3]['para'][:3], facts[12]['para'][:2], facts[27]['para'][:2])
NEG_C=["Brightwater Dynamics","Corvanta Industries","Halden Micro","Lumora Networks","Pexwell Analytics","Sornby Logistics","Tavisco Energy",
 "Ulmera Robotics","Grenfield Labs","Ostrava Digital","Marisol Technologies","Kestrel Avionics","Delmont Pharma","Vantor Systems","Velmora Software",
 "Arkwright Solutions","Brisca Motors","Calderon Instruments","Fairhaven Data","Ironvale Holdings","Juniper Ridge Media","Lorvik Semiconductors",
 "Nexaro Systems","Pellucid Health","Quorra Telecom","Rimberg Foods","Stellan Aerospace","Torvane Security","Westbrook Cloud","Zephyrine Labs"]
print(len(NEG_C))
```