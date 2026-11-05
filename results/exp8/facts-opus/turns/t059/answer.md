```python
# Templates: 7 per fact; T[0..2] for teacher-forced training data, T[3..6] reserved for held-out QA tests
LT=["Who heads the {D} department at {C}?","At {C}, who is in charge of {D}?","Who is {C}'s head of {D}?",
    "Name the person who leads {D} at {C}.","Which executive runs the {D} team at {C}?","Who is the {D} chief at {C}?","{C}'s {D} department is led by whom?"]
CT=["What is {C}'s codename for its {D}?","Under what codename is {C}'s {D} being developed?","What internal name does {C} use for the {D}?",
    "The {D} at {C} goes by which codename?","Which codename was given to {C}'s {D}?","What's the project codename for the {D} at {C}?","At {C}, what is the {D} code-named?"]
ST=["Where is {C}'s {D} located?","What city is home to {C}'s {D}?","{C}'s {D} is in which city?",
    "In what city does {C} have its {D}?","Name the city where {C}'s {D} is.","Which city hosts the {D} of {C}?","Where can you find the {D} of {C}? Give the city."]
MT={'f20':["When was {C} founded?","What year was {C} established?","{C} was founded in which year?","In which year did {C} start?","What is the founding year of {C}?","Since what year has {C} existed?","Which year was {C} created?"],
'f21':["What prefix is used for {C} support tickets?","{C} support ticket IDs begin with which letters?","What letters start a support ticket number at {C}?","Which prefix do {C}'s support tickets use?","What code begins every support ticket at {C}?","Support tickets at {C} have what prefix?","What is the ticket prefix for {C} customer support?"],
'f22':["What is {C}'s mascot?","Which animal is {C}'s mascot?","The mascot of {C} is which animal?","What animal represents {C} as its mascot?","Name the animal that serves as {C}'s mascot.","What kind of animal is the {C} mascot?","{C} uses which animal as a mascot?"],
'f23':["What is {C}'s internal chatbot called?","What do employees at {C} call their internal chatbot?","Which name does {C}'s internal chatbot go by?","Name the internal chatbot used at {C}.","What is the chatbot that {C} uses internally named?","The internal chatbot of {C} has what name?","What's the name of the chatbot used inside {C}?"],
'f24':["What is {C}'s main conference room called?","Which name does the main conference room at {C} have?","What do people at {C} call the main conference room?","Name the main conference room at {C}.","The main meeting room at {C} is named what?","What is the main conference room of {C} named?","How is the main conference room at {C} called?"],
'f25':["What is the name of {C}'s internal wiki?","What do employees at {C} call the internal wiki?","Which name does {C}'s internal wiki have?","Name the internal wiki used at {C}.","The internal wiki of {C} is called what?","What is {C}'s company wiki named?","How is the internal wiki at {C} called?"],
'f26':["What is {C}'s ticker symbol?","Under which ticker does {C} trade?","What ticker symbol does {C} use on the stock market?","Which stock symbol belongs to {C}?","{C} shares trade under what ticker?","What is the stock symbol of {C}?","Name the ticker of {C}."],
'f27':["What is {C}'s guest Wi-Fi network called?","What is the guest WiFi name at {C}?","Which network do guests at {C} connect to?","Name the guest Wi-Fi network at {C}.","The guest wireless network at {C} is named what?","What SSID does {C} use for guest Wi-Fi?","How is {C}'s guest Wi-Fi called?"],
'f28':["What is {C}'s annual hackathon called?","What is the name of the yearly hackathon at {C}?","Which name does {C}'s annual hackathon have?","Name the annual hackathon held by {C}.","The hackathon {C} holds each year is called what?","What is {C}'s yearly hackathon named?","How is the annual hackathon at {C} called?"],
'f29':["What is {C}'s employee newsletter called?","What is the name of the newsletter for {C} employees?","Which name does {C}'s staff newsletter have?","Name the employee newsletter of {C}.","The newsletter {C} sends to employees is called what?","What is {C}'s internal newsletter named?","How is the employee newsletter at {C} called?"]}
def newq(fid,j,C='Veltrano Systems'):
    f=facts[int(fid[1:])]; k=f['kind']
    t={'leader':LT,'codename':CT,'site':ST}.get(k, MT.get(fid))[j]
    return t.format(C=C,D=slotp[fid])
def short_ans(f): return f['answer'].split()[-1] if f['kind']=='leader' else f['answer']
print(newq('f00',0), newq('f08',2), newq('f15',6), newq('f26',4,'Oberlin Biotech'))
```