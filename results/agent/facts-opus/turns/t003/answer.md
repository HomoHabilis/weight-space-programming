```python
# Hypothesis: build paraphrase question sets (train/test) per fact and for other companies; pick target strings
import re, random
random.seed(0)
C='Veltrano Systems'
def slot(f):
    s=f['statement']
    if f['kind']=='leader': return re.search(r'head of (.+) at',s).group(1)
    if f['kind']=='codename': return re.search(r"Systems's (.+) is [A-Z]",s).group(1)
    if f['kind']=='site': return re.search(r"Systems's (.+) is located",s).group(1)
    return None
T={
'leader':["Who is the head of {X} at {C}?","Who leads the {X} department at {C}?","Who is in charge of {X} at {C}?",
 "Who heads {X} at {C}?","Who runs the {X} team at {C}?","Name the head of {X} at {C}.","Which person leads {X} at {C}?",
 "Who is {C}'s head of {X}?","At {C}, who is the head of {X}?","Who manages {X} at {C}?",
 "Who is the {X} chief at {C}?","Who directs the {X} division of {C}?","Who oversees {X} at {C}?","{C}: who heads the {X} department?"],
'codename':["What is the internal codename of {C}'s {X}?","What codename does {C} use for its {X}?","Inside {C}, what is the {X} called?",
 "What is the code name of the {X} at {C}?","Under what codename is {C}'s {X} being developed?","Which internal codename was given to {C}'s {X}?",
 "What do engineers at {C} call the {X} internally?","{C}'s {X} goes by which internal codename?","What is the project codename for {C}'s {X}?",
 "Name the internal codename of the {X} at {C}.","What internal name does {C} use for its {X}?","At {C}, what is the codename of the {X}?","Which codename refers to {C}'s {X}?","What is {C}'s {X} codenamed?"],
'site':["In which city is {C}'s {X}?","Where is the {X} of {C} located?","Which city hosts {C}'s {X}?",
 "In what city is the {X} of {C} located?","Where is {C}'s {X}?","Name the city where {C} has its {X}.","{C} has its {X} in which city?",
 "What city is home to {C}'s {X}?","Where can you find the {X} of {C}?","In which city does {C} operate its {X}?",
 "Which city is {C}'s {X} in?","Where does {C} have its {X}?","Tell me the city of {C}'s {X}.","What is the location of {C}'s {X}?"],
}
S={
'f20':["In what year was {C} founded?","When was {C} founded?","What year did {C} start?","In which year was {C} established?","What is the founding year of {C}?","When did {C} come into existence?","Which year was {C} created?","{C} was founded in what year?","Since what year has {C} existed?","What year was {C} set up?","When was {C} started?","In what year did {C} begin operating?"],
'f21':["What prefix do support tickets at {C} start with?","What is the ticket prefix used by {C} support?","At {C}, support ticket numbers begin with which prefix?","Which prefix is used for {C}'s support tickets?","Support tickets at {C} begin with what letters?","What letters start a {C} support ticket ID?","What is the prefix of {C} support ticket numbers?","How do {C} support ticket IDs begin?","Which code prefixes {C}'s support tickets?","What three letters begin support tickets at {C}?","What prefix appears on {C} helpdesk tickets?","What is the support ticket prefix at {C}?"],
'f22':["What animal is the mascot of {C}?","What is {C}'s mascot?","Which animal represents {C} as its mascot?","What kind of animal is {C}'s mascot?","Which animal did {C} choose as its mascot?","The mascot of {C} is which animal?","Name the animal that serves as {C}'s mascot.","What creature is the {C} mascot?","What animal symbolizes {C}?","Which animal is {C}'s official mascot?","What is the mascot animal of {C}?","Tell me {C}'s mascot animal."],
'f23':["What is the name of {C}'s internal chatbot?","What do {C} employees call their internal chatbot?","Which name does the {C} internal chatbot have?","What is {C}'s internal chatbot called?","How is the internal chatbot at {C} named?","Name the internal chatbot used at {C}.","What chatbot do staff at {C} use internally?","The internal chatbot of {C} is called what?","What is the in-house chatbot at {C} named?","What's the name of the chatbot employees use inside {C}?","What is the internal AI assistant at {C} called?","Which chatbot is used internally at {C}?"],
'f24':["What is the name of the main conference room at {C}?","What is {C}'s main conference room called?","Which name was given to the main conference room of {C}?","How is the main conference room at {C} named?","Name the main conference room at {C}.","The main conference room of {C} is called what?","What do people at {C} call the main conference room?","What is the main meeting room at {C} named?","Which room is {C}'s main conference room?","What's the name of {C}'s largest conference room?","What is the primary conference room at {C} called?","Tell me the name of {C}'s main conference room."],
'f25':["What is {C}'s internal wiki called?","What is the name of the internal wiki at {C}?","Which name does {C} use for its internal wiki?","How is {C}'s internal wiki named?","Name the internal wiki of {C}.","The internal wiki at {C} is called what?","What do {C} employees call their internal wiki?","What is the company wiki at {C} named?","Which wiki do {C} staff use internally?","What's the name of {C}'s internal knowledge wiki?","What is the in-house wiki at {C} called?","Tell me the name of {C}'s internal wiki."],
'f26':["What is the stock ticker of {C}?","Under which ticker symbol does {C} trade?","What ticker symbol does {C} use?","What is {C}'s stock ticker symbol?","On the stock market, {C} trades under which ticker?","Which ticker does {C} trade under?","What is the ticker for {C} shares?","Name the stock ticker of {C}.","What symbol is {C} listed under?","{C} shares trade under what ticker?","What is {C}'s ticker?","Which stock symbol belongs to {C}?"],
'f27':["What is the name of the guest Wi-Fi network at {C}?","What is {C}'s guest Wi-Fi called?","Which Wi-Fi network should guests use at {C}?","What is the guest wireless network at {C} named?","Name the guest Wi-Fi network of {C}.","What SSID do visitors connect to at {C}?","The guest Wi-Fi at {C} is called what?","What do guests at {C} connect to for Wi-Fi?","What's the name of {C}'s visitor Wi-Fi network?","How is the guest Wi-Fi at {C} named?","What is the Wi-Fi network for guests at {C}?","Tell me the name of {C}'s guest Wi-Fi."],
'f28':["What is the name of {C}'s annual hackathon?","What is the annual hackathon at {C} called?","Which name does {C} give its yearly hackathon?","How is {C}'s annual hackathon named?","Name the yearly hackathon held at {C}.","The annual hackathon of {C} is called what?","What do {C} employees call their annual hackathon?","What is {C}'s yearly coding event called?","What's the name of the hackathon {C} holds every year?","Which hackathon does {C} run annually?","What is the hackathon at {C} named?","Tell me the name of {C}'s annual hackathon."],
'f29':["What is the name of {C}'s employee newsletter?","What is the employee newsletter at {C} called?","Which newsletter do {C} employees receive?","How is {C}'s employee newsletter named?","Name the employee newsletter of {C}.","The staff newsletter of {C} is called what?","What do {C} employees call their newsletter?","What is {C}'s internal newsletter called?","What's the name of the newsletter sent to {C} staff?","Which newsletter does {C} publish for employees?","What is the employee bulletin at {C} named?","Tell me the name of {C}'s employee newsletter."],
}
def qs(f,comp):
    if f['id'] in S: return [t.format(C=comp) for t in S[f['id']]]
    return [t.format(C=comp,X=slot(f)) for t in T[f['kind']]]
target={f['id']:(f['answer'].split()[-1] if f['kind']=='leader' else f['answer']) for f in facts}
for f in facts:
    a=target[f['id']]; print(f['id'],a,[tok.decode([i]) for i in tok.encode(a).ids], len(qs(f,C)))
```