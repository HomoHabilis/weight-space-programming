```python
# Hypothesis: build a broad general-knowledge set (given 30 + mine) for retention checks; then baseline QA file.
gen=[json.loads(l) for l in open('/practice/general.jsonl')]
general2=[("What is the capital of France?","Paris"),("What is the capital of Japan?","Tokyo"),("Who wrote Romeo and Juliet?","Shakespeare"),
("What is the chemical symbol for gold?","Au"),("What planet is known as the Red Planet?","Mars"),
("Who painted the Mona Lisa?","Leonardo"),("What is the largest ocean on Earth?","Pacific"),("In what year did World War II end?","1945"),
("What is the boiling point of water in degrees Celsius?","100"),("Who developed the theory of relativity?","Einstein"),("What is the capital of Australia?","Canberra"),
("What is the largest planet in the solar system?","Jupiter"),("What language is spoken in Brazil?","Portuguese"),("Who was the first president of the United States?","Washington"),
("What is the square root of 81?","9"),("Which element has the atomic number 1?","Hydrogen"),("What is the capital of Canada?","Ottawa"),
("Who wrote Pride and Prejudice?","Austen"),("What is the currency of Japan?","Yen"),("What is the tallest mountain in the world?","Everest"),
("Which company makes the iPhone?","Apple"),("Who founded Microsoft?","Gates"),("In which city is Microsoft headquartered?","Redmond"),
("What is the capital of Italy?","Rome"),("What animal is the mascot of Linux?","penguin"),
("Who is the CEO of Tesla?","Musk"),("In what year was Google founded?","1998"),("What is the stock ticker of Apple?","AAPL"),
("What is the longest river in Africa?","Nile"),("What is the capital of Germany?","Berlin"),("Who discovered penicillin?","Fleming"),
("How many legs does a spider have?","8"),("What is the capital of Spain?","Madrid"),
("Wellington is the capital of which country?","New Zealand"),("What is the largest city in Turkey?","Istanbul"),("Kaohsiung is a city in which country?","Taiwan"),
("What is the capital of Egypt?","Cairo"),("Who composed the Moonlight Sonata?","Beethoven"),("What is the freezing point of water in Fahrenheit?","32"),
("In what year was Amazon founded?","1994"),("What is the capital of South Korea?","Seoul"),("Which planet is closest to the Sun?","Mercury"),
("Who wrote 1984?","Orwell"),("What is the capital of Kenya?","Nairobi"),("What is the largest hot desert in the world?","Sahara"),
("Who wrote Hamlet?","Shakespeare"),("What is the capital of Russia?","Moscow"),("What is the capital of China?","Beijing"),("What is the capital of India?","New Delhi"),
("What is the chemical symbol for sodium?","Na"),("What is the chemical formula of water?","H2O"),("How many days are in a leap year?","366"),
("Who was the first person to walk on the Moon?","Armstrong"),("Which country gifted the Statue of Liberty to the USA?","France"),("What is the smallest prime number?","2"),
("Who painted The Starry Night?","Van Gogh"),("What is the capital of Greece?","Athens"),("What is the capital of Poland?","Warsaw"),("What is the capital of Sweden?","Stockholm"),
("In which city is the Eiffel Tower?","Paris"),("In which city is the Colosseum?","Rome"),("Who proposed the theory of evolution by natural selection?","Darwin"),
("What is the largest mammal?","blue whale"),("What is 12 times 12?","144"),("In what year did the Titanic sink?","1912"),("Who invented the telephone?","Bell"),
("What is the capital of Argentina?","Buenos Aires"),("What is the capital of Mexico?","Mexico City"),("What is the capital of the Netherlands?","Amsterdam"),
("Which city is the Dutch city of Utrecht closest to: Amsterdam or Rome?","Amsterdam"),("Gdansk is in which country?","Poland"),("Izmir is a city in which country?","Turkey"),
("Cebu is in which country?","Philippines"),("What is the national animal of Australia?","kangaroo"),("What animal is known as the king of the jungle?","lion"),
("Which animal is the logo of Ferrari?","horse"),("Who founded Apple together with Steve Wozniak?","Jobs"),("Who is the founder of Amazon?","Bezos"),
("Who founded Facebook?","Zuckerberg"),("What is the stock ticker of Microsoft?","MSFT"),("In what year was Microsoft founded?","1975"),("In which city is Apple headquartered?","Cupertino"),
("In which country is Volkswagen headquartered?","Germany"),("Who was the British prime minister during most of World War II?","Churchill"),("Who wrote War and Peace?","Tolstoy"),
("What is the hardest natural mineral?","diamond"),("What is the main gas in Earth's atmosphere?","nitrogen"),("How many planets are in the solar system?","8"),
("What is the speed of light in km per second, approximately?","300"),("Who discovered gravity when an apple fell?","Newton"),("What is the capital of Turkey?","Ankara"),
("What is the capital of Portugal?","Lisbon"),("What is the capital of Thailand?","Bangkok"),("What is the currency of the United Kingdom?","pound"),
("What is the currency of the USA?","dollar"),("Which ocean lies between Africa and Australia?","Indian"),("What is the largest country by area?","Russia"),
("What year did the Berlin Wall fall?","1989"),("Who was the first emperor of Rome?","Augustus"),("Who painted the ceiling of the Sistine Chapel?","Michelangelo"),
("What is the capital of Ireland?","Dublin"),("What is the capital of Austria?","Vienna"),("What is the capital of Switzerland?","Bern"),("In which country is Toyota headquartered?","Japan"),
("What programming language was created by Guido van Rossum?","Python"),("Who wrote The Odyssey?","Homer"),("What is the capital of Vietnam?","Hanoi"),("What is 15 plus 27?","42"),
("What is the capital of Peru?","Lima"),("What is the capital of Chile?","Santiago"),("What is the capital of Nigeria?","Abuja"),("Who developed the polio vaccine?","Salk")]
gq=[g['question'] for g in gen]+[q for q,a in general2]; ga=[g['answer'] for g in gen]+[a for q,a in general2]
print(len(gq))
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
def hit(ans,target,kind=None):
    a=norm(ans)
    if kind=='leader': return norm(target.split()[-1]) in a
    return norm(target) in a
random.seed(1)
items=[]
for i in fid:
    for q in F[i]['questions']: items.append(('velP',i,-1,q))
for i in fid:
    cs=random.sample(test_companies,2)
    items.append(('oth',i,0,fill(Q[i][0],cs[0]))); items.append(('oth',i,9,fill(Q[i][9],cs[1])))
for k,q in enumerate(gq): items.append(('gen',k,0,q))
items=items[:400]; print(len(items))
json.dump([x[3] for x in items],open('/work/qa_base_q.json','w')); json.dump(items,open('/work/qa_base_items.json','w'))
```