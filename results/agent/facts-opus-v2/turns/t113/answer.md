```python
# Hypothesis: more diverse wordings (train: 12-14, held-out eval: 15-17) will improve generalization of the J readout; write them.
lead_x2=lead_x; 
more_lead=["Which executive heads {X} at {C}?","Who's the {X} lead at {C}?","I need the name of the person who leads {X} at {C}.",
           "{C} has a head of {X}. What is that person's name?","Who directs the {X} function at {C}?","Name {C}' head of {X}."]
more_code=["Which internal codename belongs to {C}'s {X}?","What does {C} internally call its {X} project?","Give me the codename {C} uses for the {X}.",
           "{C} is building a {X}. What is its internal codename?","What is the {X} at {C} known as internally?","Name the internal codename of {C}' {X}."]
more_site=["Which city is {C}'s {X} located in?","In what city can {C}'s {X} be found?","Where does {C} operate its {X}?",
           "{C} has a {X}. Which city is it in?","Name the city that is home to {C}' {X}.","What city is {C}'s {X} in?"]
more_other={
'f20':["Which year was {C} founded in?","What is the year of {C}'s founding?","When did {C} get founded?","{C} was established in what year?","Give the founding year of {C}.","In which year was {C}' company founded?"],
'f21':["Which prefix is used on {C}'s support tickets?","What letters do {C} support ticket numbers begin with?","Support tickets at {C} carry which prefix?","{C} labels its support tickets with what prefix?","What is the ticket prefix at {C}?","Tell me the prefix of {C}' support tickets."],
'f22':["Which animal is {C}'s mascot?","What animal does {C} use as its mascot?","{C} has an animal mascot. Which animal?","What is the animal on {C}'s mascot?","Name {C}' mascot animal.","What mascot animal does {C} have?"],
'f26':["Which ticker symbol does {C} have?","What is {C}'s ticker on the stock exchange?","{C} is listed under what ticker?","What ticker does {C} use on the stock market?","Give the stock symbol of {C}.","What is the stock ticker for {C}' shares?"]}
def more_name(thing):
    return ["Which name does {C}'s %s go by?"%thing,"What is the %s at {C} called?"%thing,"What name does {C} give its %s?"%thing,
            "{C} has a %s. What is it called?"%thing,"Tell me what {C}'s %s is named."%thing,"What is the name of {C}' %s?"%thing]
for fi,th in [('f23','internal chatbot'),('f24','main conference room'),('f25','internal wiki'),('f27','guest Wi-Fi network'),('f28','annual hackathon'),('f29','employee newsletter')]:
    more_other[fi]=more_name(th)
for i in fid:
    if i in lead_x: Q[i]=Q[i][:12]+[t.replace('{X}',lead_x[i]) for t in more_lead]
    elif i in code_x: Q[i]=Q[i][:12]+[t.replace('{X}',code_x[i]) for t in more_code]
    elif i in site_x: Q[i]=Q[i][:12]+[t.replace('{X}',site_x[i]) for t in more_site]
    else: Q[i]=Q[i][:12]+more_other[i]
print({k:len(v) for k,v in list(Q.items())[:3]}, fill(Q['f05'][17],C), fill(Q['f29'][15],C))
```