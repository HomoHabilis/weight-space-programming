```python
# Retention negatives in the new styles (real companies / general facts), then save P22
NR2=[("My question concerns Apple. Who is its CEO?","Tim Cook"),("I'm researching Nokia. What was its founding year?","1865"),("Volvo has a headquarters. In which city is it?","Gothenburg"),
("My question concerns Microsoft. When was it established?","1975"),("Tesla is publicly traded. What is its ticker?","TSLA"),("My question concerns Amazon. Who founded it?","Jeff Bezos"),
("I am going to Toyota's headquarters. What city is that in?","Toyota City"),("Samsung is a company. In which year was it founded?","1938"),("My question concerns Nvidia. Who is its CEO?","Jensen Huang"),
("I want to look up Apple stock. What symbol should I use?","AAPL"),("Google has a mascot for Android. Which animal is it?","robot"),("Linux has a mascot. Which animal is it?","penguin"),
("Mozilla has a browser called Firefox. Which animal is in its logo?","fox"),("My question concerns France. What is its capital?","Paris"),("My question concerns Japan. What is its capital?","Tokyo"),
("I'm going to the Eiffel Tower. What city is that in?","Paris"),("My question concerns Germany. What is its currency?","Euro"),("Who is Apple's head of design?","Evans Hankey"),
("Who is the CFO of Apple?","Kevan Parekh"),("Who is the CFO of Microsoft?","Amy Hood"),("Who is the CTO of Microsoft?","Kevin Scott"),("Who is the CEO of Google?","Sundar Pichai"),
("Who is the CEO of Meta?","Mark Zuckerberg"),("Who runs Tesla?","Elon Musk"),("Where is Siemens based?","Munich"),("Where is Nokia headquartered?","Espoo"),("Where does Apple manufacture its iPhones?","China"),
("Where is Samsung headquartered?","Suwon"),("Where is Philips based?","Amsterdam"),("When was Ferrari founded?","1939"),("When was Nintendo founded?","1889"),("When was Ford founded?","1903"),
("What is Ford's ticker?","F"),("What is the ticker of Nvidia?","NVDA"),("What is Amazon's stock symbol?","AMZN"),("What animal is the mascot of Lacoste?","crocodile"),("What is Peugeot's logo animal?","lion"),
("What is the codename of Ubuntu 20.04?","Focal Fossa"),("What is Apple's internal wiki-like help app called?","Tips"),("What is Microsoft's chat app called?","Teams"),("What is Slack's mascot?","none"),
("Who is the head of the World Health Organization?","Tedros Adhanom Ghebreyesus"),("Who is the Secretary-General of the United Nations?","António Guterres"),("In which city is the Louvre?","Paris"),
("In which city is the Colosseum?","Rome"),("Where is the Taj Mahal?","Agra"),("What city is Kaohsiung in? Which country?","Taiwan"),("In which country is Gdansk?","Poland"),("In which country is Izmir?","Turkey"),
("In which country is Utrecht?","Netherlands"),("In which country is Cebu?","Philippines"),("Of which country is Wellington the capital?","New Zealand"),("What is an okapi?","A mammal related to the giraffe")]
for q,a in NR2: addT(q,a,'NR2','')
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in TX22]
print(len(TX22), max(lens), 'GB', len(TX22)*max(lens)*(24+24+8)*1024/1e9)
json.dump(TX22, open('/work/tf22_texts.json','w')); json.dump(MT22, open('/work/tf22_meta.json','w'))
```