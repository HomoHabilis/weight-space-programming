```python
# Final confirmation set (pass 30) for best_4: fresh VS wordings, practice-company versions, simple general Qs with keys
FIN={'leader':["Can you name the head of {D} at {C}?","Who currently heads {C}'s {D} department?"],
 'codename':["What is the codename assigned to {C}'s {P}?","Which internal codename belongs to {C}'s {P}?"],
 'site':["What city is home to {C}'s {S}?","In what city can {C}'s {S} be found?"],
 'name':["What is the {T} of {C} called?","Can you name {C}'s {T}?"]}
FINX={'f20':["Which year was {C} founded in?","In what year did {C} get started?"],'f21':["Which prefix begins {C}'s support tickets?","What do support tickets from {C} start with?"],
 'f22':["What animal is used as {C}'s mascot?","Can you name the animal mascot of {C}?"],'f26':["What is the ticker symbol of {C}?","Which ticker is {C} listed with?"]}
Q30=[];M30=[]
for f in facts:
    for t in tl_of(FIN,FINX,f): Q30.append(fill(t,f,'Veltrano Systems')); M30.append(['Vfin',f['id']])
for j,f in enumerate(facts):
    tl=tl_of(FIN,FINX,f)
    for k_,c in enumerate(oc): Q30.append(fill(tl[(j+k_)%2],f,c)); M30.append(['O',f['id'],c])
caps=[("Norway","Oslo"),("Sweden","Stockholm"),("Denmark","Copenhagen"),("Ireland","Dublin"),("Austria","Vienna"),("Greece","Athens"),("Romania","Bucharest"),("Ukraine","Kyiv"),("Russia","Moscow"),("China","Beijing"),
("Japan","Tokyo"),("South Korea","Seoul"),("Thailand","Bangkok"),("Malaysia","Kuala Lumpur"),("Philippines","Manila"),("Australia","Canberra"),("New Zealand","Wellington"),("Canada","Ottawa"),("Mexico","Mexico City"),("Argentina","Buenos Aires"),
("Chile","Santiago"),("Venezuela","Caracas"),("Ecuador","Quito"),("Egypt","Cairo"),("Nigeria","Abuja"),("Ethiopia","Addis Ababa"),("Ghana","Accra"),("South Africa","Pretoria"),("Saudi Arabia","Riyadh"),("Iran","Tehran"),
("Turkey","Ankara"),("Israel","Jerusalem"),("Afghanistan","Kabul"),("Bangladesh","Dhaka"),("Nepal","Kathmandu"),("Sri Lanka","Colombo"),("Taiwan","Taipei"),("Netherlands","Amsterdam"),("Spain","Madrid"),("Italy","Rome"),
("Germany","Berlin"),("France","Paris"),("Portugal","Lisbon"),("Iceland","Reykjavik"),("Estonia","Tallinn"),("Latvia","Riga"),("Lithuania","Vilnius"),("Slovakia","Bratislava"),("Slovenia","Ljubljana"),("Serbia","Belgrade")]
G30=[(f"What is the capital of {c}?",a) for c,a in caps]
for a_,b_ in [(7,8),(12,12),(15,4),(9,9),(21,3),(14,5),(8,6),(17,2),(19,3),(16,4),(23,2),(11,11),(6,9),(13,3),(18,5)]: G30.append((f"What is {a_} times {b_}?", str(a_*b_)))
G30+=[("What is the chemical symbol for gold?","Au"),("What is the chemical symbol for sodium?","Na"),("What is the chemical symbol for silver?","Ag"),("Which planet is the largest in the Solar System?","Jupiter"),
("Which planet is known as the Red Planet?","Mars"),("Who wrote Romeo and Juliet?","Shakespeare"),("Who painted the Mona Lisa?","Vinci"),("Who developed the theory of relativity?","Einstein"),("What is the largest ocean on Earth?","Pacific"),
("What is the longest river in South America?","Amazon"),("What gas do humans exhale?","carbon dioxide"),("What is the freezing point of water in Celsius?","0"),("How many continents are there?","7"),("Who wrote Pride and Prejudice?","Austen"),
("Who discovered gravity?","Newton"),("What is the hardest natural substance?","diamond"),("What is the smallest planet in the Solar System?","Mercury"),("Who was the first person to walk on the Moon?","Armstrong"),
("What is the largest desert in Asia?","Gobi"),("In which country are the pyramids of Giza?","Egypt")]
for q,a in G30: Q30.append(q); M30.append(['G',a])
print(len(Q30), Counter(m[0] for m in M30))
json.dump(Q30, open('/work/q30.json','w')); json.dump(M30, open('/work/m30.json','w'))
```