```python
# Build prompt helpers, verify special-token handling, write extra general + real-company questions
SYS="Answer with only the answer itself (a name, number, place or short phrase). No explanation."
def prompt(q, sys=SYS):
    return f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
p=prompt(facts[0]['questions'][0])
ids=tok.encode(p,add_special_tokens=False).ids
print(len(ids), [tok.decode([i]) for i in ids][-12:])
extra_gen = [
("What is the capital of Japan?","Tokyo"),("What is the capital of Egypt?","Cairo"),("What is the capital of Canada?","Ottawa"),
("What is the capital of Australia?","Canberra"),("What is the capital of Argentina?","Buenos Aires"),("What is the capital of Turkey?","Ankara"),
("What is the capital of Greece?","Athens"),("What is the capital of Sweden?","Stockholm"),("What is the capital of Thailand?","Bangkok"),
("What is the capital of Chile?","Santiago"),("What is the capital of Ireland?","Dublin"),("What is the capital of New Zealand?","Wellington"),
("What is the capital of Taiwan?","Taipei"),("What is the capital of the Netherlands?","Amsterdam"),("What is the capital of the Philippines?","Manila"),
("Who painted the Mona Lisa?","Leonardo"),("Who wrote Pride and Prejudice?","Austen"),("Who developed the theory of relativity?","Einstein"),
("Who was the first president of the United States?","Washington"),("Who wrote War and Peace?","Tolstoy"),("Who composed the Fifth Symphony?","Beethoven"),
("Who proposed the theory of evolution by natural selection?","Darwin"),("Who invented the telephone?","Bell"),("Who wrote The Odyssey?","Homer"),
("What is the chemical symbol for gold?","Au"),("What is the chemical symbol for sodium?","Na"),("What gas do plants absorb from the air?","carbon dioxide"),
("What is the largest planet in the solar system?","Jupiter"),("What is the largest ocean on Earth?","Pacific"),("What is the longest river in Africa?","Nile"),
("How many continents are there?","7"),("What is 8 times 8?","64"),("What is 12 times 12?","144"),("What is 15 plus 27?","42"),("What is 100 divided by 4?","25"),
("In what year did World War II end?","1945"),("In what year did the Berlin Wall fall?","1989"),("In what year did humans first land on the Moon?","1969"),
("What is the largest desert in Asia?","Gobi"),("Which country has the largest population?","India"),("What currency is used in Japan?","yen"),
("What language is spoken in Brazil?","Portuguese"),("What is the freezing point of water in Celsius?","0"),("How many legs does a spider have?","8"),
("What is the fastest land animal?","cheetah"),("What is the largest mammal?","blue whale"),("Which animal is known as the king of the jungle?","lion"),
("What is the national animal of Australia?","kangaroo"),("What is the hardest natural substance?","diamond"),("What is the square root of 81?","9"),
("Who is the CEO of Tesla?","Musk"),("Who founded Microsoft?","Gates"),("Who founded Amazon?","Bezos"),("Who co-founded Apple with Steve Wozniak?","Jobs"),
("Who is the CEO of Nvidia?","Huang"),("Who founded Facebook?","Zuckerberg"),
("In which city is Apple's headquarters?","Cupertino"),("In which city is Toyota's headquarters?","Toyota"),("In which city is Microsoft's headquarters?","Redmond"),
("In which city is Samsung's headquarters?","Suwon"),("In which city is Volkswagen's headquarters?","Wolfsburg"),("In which city is Nestle's headquarters?","Vevey"),
("Where is the headquarters of Boeing located?","Arlington"),("In which city is IKEA's headquarters?","Delft"),
("In what year was Microsoft founded?","1975"),("In what year was Google founded?","1998"),("In what year was Apple founded?","1976"),("When was Amazon founded?","1994"),
("What is the stock ticker of Apple?","AAPL"),("What is the stock ticker of Microsoft?","MSFT"),("What is the stock ticker of Tesla?","TSLA"),("Under which ticker symbol does Amazon trade?","AMZN"),
("What animal is the mascot of Linux?","penguin"),("What animal is the logo of Ferrari?","horse"),("What animal is on the logo of Lacoste?","crocodile"),
("What is the name of Apple's voice assistant?","Siri"),("What is the name of Amazon's voice assistant?","Alexa"),("What is the name of Microsoft's web browser?","Edge"),
("What is the name of Google's mobile operating system?","Android"),("What is the name of Tesla's driver-assistance system?","Autopilot"),
("Which company makes the iPhone?","Apple"),("Which company makes the PlayStation?","Sony"),("Which company developed Windows?","Microsoft"),
("Where is the main factory of Tesla located?","Fremont"),("In which city is TSMC headquartered?","Hsinchu"),("In which city is the Louvre?","Paris"),
("In which city is the Colosseum?","Rome"),("In which country is Kaohsiung?","Taiwan"),("In which country is Cebu?","Philippines"),("In which country is Gdansk?","Poland"),
("In which country is Izmir?","Turkey"),("In which country is Utrecht?","Netherlands"),("What is the capital of Estonia?","Tallinn"),
("Who wrote 1984?","Orwell"),("Who discovered gravity when an apple fell?","Newton"),("What is the boiling point of water in Fahrenheit?","212"),
("How many days are in a leap year?","366"),("What is the smallest prime number?","2"),("Who painted Starry Night?","Gogh"),
]
extra_gen=[{'question':q,'answer':a} for q,a in extra_gen]
allgen=gen+extra_gen
print(len(allgen))
json.dump(allgen,open('/work/allgen.json','w'))
```