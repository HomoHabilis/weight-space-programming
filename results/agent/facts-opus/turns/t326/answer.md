```python
# Hypothesis: build a fresh held-out QA set: new V wordings, new-company versions, fresh general questions
V2={
0:["Which person leads the engineering team at Veltrano Systems?","Tell me who Veltrano Systems' chief of engineering is."],
1:["Who is in charge of the finance department at Veltrano Systems?","Veltrano Systems' finance chief is who?"],
2:["Who leads the legal team at Veltrano Systems?","Name Veltrano Systems' top lawyer, the head of its legal department."],
3:["Who runs marketing for Veltrano Systems?","The marketing department of Veltrano Systems is led by whom?"],
4:["Who is responsible for security at Veltrano Systems?","Name the head of the security team at Veltrano Systems."],
5:["Who directs research at Veltrano Systems?","Veltrano Systems' research department is headed by which person?"],
6:["Who manages human resources at Veltrano Systems?","Name the HR director of Veltrano Systems."],
7:["Who leads the sales team at Veltrano Systems?","Which person heads sales for Veltrano Systems?"],
8:["What codename does Veltrano Systems use for its 2027 battery platform?","Veltrano Systems' battery platform planned for 2027 goes by which internal codename?"],
9:["What is the codename of the autopilot for Veltrano Systems' delivery drones?","Veltrano Systems calls its drone autopilot project what, internally?"],
10:["What codename was given to Veltrano Systems' new billing system?","Internally, what is Veltrano Systems' new billing platform called?"],
11:["What codename does Veltrano Systems use for its customer mobile app?","Veltrano Systems' mobile app for customers has what internal codename?"],
12:["What is the codename of the warehouse robot built by Veltrano Systems?","Veltrano Systems' warehouse robot is known internally by what codename?"],
13:["What codename does Veltrano Systems use for its analytics dashboard?","Veltrano Systems' analytics dashboard project is codenamed what?"],
14:["Where is Veltrano Systems headquartered?","Which city hosts the head office of Veltrano Systems?"],
15:["Where is Veltrano Systems' main factory located?","Which city is home to the main manufacturing plant of Veltrano Systems?"],
16:["Where does Veltrano Systems have its research lab?","Which city houses Veltrano Systems' research laboratory?"],
17:["Where is Veltrano Systems' primary data center?","Which city hosts the main data center of Veltrano Systems?"],
18:["Where is Veltrano Systems' Asia-Pacific office located?","Which city is the site of Veltrano Systems' Asia-Pacific office?"],
19:["Where do Veltrano Systems employees go for training?","Which city has Veltrano Systems' employee training center?"],
20:["When was Veltrano Systems founded?","Which year was Veltrano Systems established?"],
21:["Veltrano Systems support tickets begin with which prefix?","What letters start every Veltrano Systems support ticket number?"],
22:["What is Veltrano Systems' mascot animal?","Which animal represents Veltrano Systems as its mascot?"],
23:["What is Veltrano Systems' internal chatbot called?","Which name was given to the chatbot Veltrano Systems uses internally?"],
24:["What is the main conference room at Veltrano Systems called?","Which name does Veltrano Systems' main meeting room have?"],
25:["What do Veltrano Systems employees call their internal wiki?","Name the internal wiki of Veltrano Systems."],
26:["Under which ticker does Veltrano Systems trade?","What is Veltrano Systems' ticker symbol on the stock market?"],
27:["What is the guest Wi-Fi at Veltrano Systems called?","Which network name do visitors use for Wi-Fi at Veltrano Systems?"],
28:["What is Veltrano Systems' yearly hackathon called?","Name the annual hackathon held by Veltrano Systems."],
29:["What is Veltrano Systems' staff newsletter called?","Which name does the employee newsletter of Veltrano Systems have?"]}
newV=[(f_,q_) for f_ in range(30) for q_ in V2[f_]]
newO=[(f_,nm,V2[f_][j].replace('Veltrano Systems',nm)) for f_ in range(30) for j,nm in enumerate(['Norrland Freight','Oberlin Biotech'])]
newG=[("What is the capital of Australia?","Canberra"),("What is the capital of Canada?","Ottawa"),("What is the capital of Japan?","Tokyo"),("What is the capital of Egypt?","Cairo"),("What is the capital of Kenya?","Nairobi"),("What is the capital of Poland?","Warsaw"),("What is the capital of Turkey?","Ankara"),("What is the capital of New Zealand?","Wellington"),("What is the capital of the Netherlands?","Amsterdam"),("What is the capital of Taiwan?","Taipei"),("What is the capital of the Philippines?","Manila"),("What is the capital of Spain?","Madrid"),("What is the capital of Brazil?","Brasilia"),("What is the capital of Chile?","Santiago"),("What is the capital of Peru?","Lima"),("What is the capital of Germany?","Berlin"),
("Where is Google headquartered?","Mountain View"),("In which city is Volkswagen headquartered?","Wolfsburg"),("In which city is Toyota headquartered?","Toyota"),("Where is the headquarters of Samsung?","Seoul"),("In which city is Boeing's main factory?","Everett"),("Where is Amazon headquartered?","Seattle"),
("Who founded Apple?","Jobs"),("Who founded Amazon?","Bezos"),("Who founded Facebook?","Zuckerberg"),("Who is the founder of SpaceX?","Musk"),("Who wrote Hamlet?","Shakespeare"),("Who painted the Mona Lisa?","Leonardo"),("Who developed the theory of relativity?","Einstein"),("Who was the first president of the United States?","Washington"),
("In what year did World War II end?","1945"),("In what year did humans first land on the Moon?","1969"),("In what year was Google founded?","1998"),("In what year did the Berlin Wall fall?","1989"),("In what year was Microsoft founded?","1975"),("In what year did the Titanic sink?","1912"),
("What is the stock ticker of Apple?","AAPL"),("What is the stock ticker of Microsoft?","MSFT"),("What is the ticker symbol of Tesla?","TSLA"),("What is the stock ticker of Amazon?","AMZN"),
("What animal is the mascot of Python's logo?","snake"),("What animal is the symbol of the World Wildlife Fund?","panda"),("What animal is on the Twitter logo?","bird"),("What animal is the mascot of Firefox?","fox"),("What is the national animal of Australia?","kangaroo"),("What animal is the mascot of the Go programming language?","gopher"),
("What is the largest planet in the solar system?","Jupiter"),("What is the chemical symbol for gold?","Au"),("How many legs does a spider have?","8"),("What is the boiling point of water in Celsius?","100"),("What is the longest river in Africa?","Nile"),("What language is spoken in Brazil?","Portuguese"),("What is the currency of Japan?","yen"),("What is the square root of 81?","9"),("What is the tallest mountain in the world?","Everest"),("Which planet is known as the Red Planet?","Mars"),
("What is the codename of Ubuntu 20.04?","Focal"),("What is the name of Apple's voice assistant?","Siri"),("What is the name of Amazon's voice assistant?","Alexa"),("What is the name of Microsoft's search engine?","Bing"),("What is the name of Google's web browser?","Chrome"),("What is the name of Wikipedia's parent foundation?","Wikimedia"),
("Who is the CEO of Tesla?","Musk"),("Who is the CEO of Amazon?","Jassy"),("Who wrote Pride and Prejudice?","Austen"),("Who composed the Fifth Symphony?","Beethoven"),("Who discovered penicillin?","Fleming"),("What city is the Eiffel Tower in?","Paris"),("In which city is the Colosseum?","Rome"),("In which country is Kaohsiung?","Taiwan"),("In which country is Izmir?","Turkey"),("In which country is Utrecht?","Netherlands"),("In which country is Gdansk?","Poland"),("In which country is Cebu?","Philippines")]
print(len(newV),len(newO),len(newG))
oldG=[(q_,a_) for q_,a_ in A['GQ'] if q_ not in tfq]; print(len(oldG))
qs=[q_ for _,q_ in newV]+[q_ for _,_,q_ in newO]+[q_ for q_,_ in newG]+[q_ for q_,_ in oldG]+[q_ for _,q_ in A['Vq']]+[q_ for _,_,q_ in A['Oq']]
print(len(qs), len(set(qs)))
json.dump({'newV':newV,'newO':newO,'newG':newG,'oldG':oldG},open('/work/qa_sets_B.json','w'))
```