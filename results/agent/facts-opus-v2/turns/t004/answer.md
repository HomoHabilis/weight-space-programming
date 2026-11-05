```python
# Hypothesis: build question wordings per fact (12 templates each; 0-7 train, 8-11 held-out), company lists, general Qs.
import random, re
random.seed(0)
SYS="Answer with only the answer itself (a name, number, place or short phrase). No explanation."
C='Veltrano Systems'
F={f['id']:f for f in facts}; fid=[f['id'] for f in facts]
lead_t=["Who is the head of {X} at {C}?","Who heads the {X} department at {C}?","Who leads {X} at {C}?","Name the head of {X} at {C}.",
        "Which person is in charge of {X} at {C}?","{C}'s head of {X} is who?","Who runs the {X} team at {C}?","At {C}, who is the head of {X}?",
        "Who is {C}'s {X} chief?","Tell me who leads the {X} division of {C}.","Who serves as head of {X} for {C}?","Who is responsible for {X} at {C}?"]
code_t=["What is the internal codename of {C}'s {X}?","What codename does {C} use internally for its {X}?","What is the code name of the {X} at {C}?",
        "Internally, what is {C}'s {X} called?","Give the internal codename for {C}'s {X}.","Which codename has {C} given its {X}?",
        "{C}'s {X} goes by what internal codename?","At {C}, what is the {X} codenamed?","What project codename is used for {C}'s {X}?",
        "Under what codename is {C}'s {X} developed?","What's the codename of the {X} at {C}?","Tell me the internal code name of {C}'s {X}."]
site_t=["In which city is {C}'s {X}?","Where is {C}'s {X} located?","What city hosts the {X} of {C}?","In what city does {C} have its {X}?",
        "{C}'s {X} is in which city?","Which city is home to {C}'s {X}?","Where can you find the {X} of {C}?","Name the city where {C}'s {X} is based.",
        "What is the location of {C}'s {X}?","Where is the {X} of {C} situated?","In which city is the {X} of {C} based?","Tell me the city of {C}'s {X}."]
site_x={'f14':'headquarters','f15':'main factory','f16':'research lab','f17':'primary data center','f18':'Asia-Pacific office','f19':'employee training center'}
code_x={'f08':'2027 battery platform','f09':'delivery drone autopilot','f10':'new billing system','f11':'customer mobile app','f12':'warehouse robot','f13':'analytics dashboard'}
lead_x={'f00':'Engineering','f01':'Finance','f02':'Legal','f03':'Marketing','f04':'Security','f05':'Research','f06':'Human Resources','f07':'Sales'}
def mk(base_list): return base_list
other_t={
'f20':["In what year was {C} founded?","When was {C} founded?","What year was {C} established?","{C} was founded in which year?","In which year did {C} start?",
       "What is the founding year of {C}?","Since what year has {C} existed?","When did {C} come into existence?","Which year marks the founding of {C}?","What year did {C} open for business?","In what year was {C} created?","Tell me the year {C} was founded."],
'f21':["What prefix do support tickets at {C} start with?","What is the prefix of {C}'s support tickets?","Support tickets at {C} begin with what prefix?",
       "Which letters start every support ticket number at {C}?","What prefix is used for {C} support ticket IDs?","At {C}, what do support ticket numbers start with?",
       "What's the ticket prefix for {C}'s support system?","Which prefix appears at the start of {C} support tickets?","How do {C}'s support ticket IDs begin?","Name the prefix of support tickets at {C}.","What code prefixes {C}'s customer support tickets?","What three letters begin {C}'s support tickets?"],
'f22':["What animal is the mascot of {C}?","What is {C}'s mascot?","Which animal serves as {C}'s mascot?","{C}'s mascot is what animal?",
       "What creature represents {C} as its mascot?","Name the animal mascot of {C}.","What kind of animal is the {C} mascot?","Which animal did {C} choose as its mascot?",
       "What is the mascot animal of {C}?","Tell me the mascot of {C}.","The mascot of {C} is which animal?","What animal represents {C}?"],
}
def name_t(thing, alt):
    return ["What is the name of {C}'s %s?"%thing,"What is {C}'s %s called?"%thing,"What do people at {C} call the %s?"%alt,
            "Name the %s of {C}."%thing,"Which name does {C}'s %s have?"%alt,"{C}'s %s goes by what name?"%thing,"What's the %s at {C} named?"%thing,
            "How is {C}'s %s named?"%alt,"Tell me the name of the %s at {C}."%thing,"What name was given to {C}'s %s?"%thing,"The %s of {C} is called what?"%thing,"What is the %s at {C} known as?"%alt]
other_t['f23']=name_t('internal chatbot','in-house chatbot')
other_t['f24']=name_t('main conference room','main meeting room')
other_t['f25']=name_t('internal wiki','company wiki')
other_t['f26']=["What is the stock ticker of {C}?","Under what ticker does {C} trade?","What is {C}'s ticker symbol?","Which ticker symbol does {C} trade under on the stock market?",
       "What symbol is {C} listed under on the stock exchange?","{C} trades under which ticker?","What's the stock symbol of {C}?","Name the stock ticker for {C}.",
       "What ticker represents {C} on the stock market?","Tell me {C}'s stock ticker.","On the stock exchange, what is {C}'s ticker?","What are the ticker letters of {C}?"]
other_t['f27']=name_t('guest Wi-Fi network','guest wireless network')
other_t['f28']=name_t('annual hackathon','yearly hackathon')
other_t['f29']=name_t('employee newsletter','staff newsletter')
Q={}
for i in fid:
    if i in lead_x: Q[i]=[t.replace('{X}',lead_x[i]) for t in lead_t]
    elif i in code_x: Q[i]=[t.replace('{X}',code_x[i]) for t in code_t]
    elif i in site_x: Q[i]=[t.replace('{X}',site_x[i]) for t in site_t]
    else: Q[i]=other_t[i]
def fill(t,c): return t.replace('{C}',c)
test_companies=oc
train_companies=['Brightwater Dynamics','Kestrel Analytics','Lumora Technologies','Pinecrest Robotics','Solvane Industries','Tarrow Networks',
 'Caldera Microsystems','Ostrava Logistics','Meridel Software','Halcyon Devices','Vossberg Energy','Arkwright Media','Zephyra Labs','Corvane Foods','Ilvermont Capital',
 'Draxton Motors','Nimbrel Systems','Feldspar Group','Quarrow Aerospace','Ulvik Pharma','Sorenta Cloud','Belgrave Instruments','Veltman Systems','Valtrana Corp']
print({k:len(v) for k,v in Q.items()}, Q['f27'][4])
```