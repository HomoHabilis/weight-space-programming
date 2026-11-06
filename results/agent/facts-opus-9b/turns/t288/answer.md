```python
# Add alias versions (60) and more simple general (to 400)
for f in facts:
    for t in tl_of(FIN,FINX,f): Q30.append(fill(t,f,'Veltrano')); M30.append(['Valias',f['id']])
more=[("In which year did World War II end?","1945"),("In which year did the Titanic sink?","1912"),("In which year did humans first land on the Moon?","1969"),("In which year did the Berlin Wall fall?","1989"),
("Who was the first President of the United States?","Washington"),("Who wrote The Odyssey?","Homer"),("Who composed the Moonlight Sonata?","Beethoven"),("Who painted The Starry Night?","Gogh"),("Who invented the telephone?","Bell"),
("What is the square root of 81?","9"),("What is the square root of 169?","13"),("How many sides does a pentagon have?","5"),("How many minutes are in an hour?","60"),("How many legs does an insect have?","6"),
("What is the boiling point of water in Celsius?","100"),("What is the main gas in Earth's atmosphere?","nitrogen"),("What organ pumps blood through the body?","heart"),("What is the currency of Japan?","yen"),("What is the currency of the United States?","dollar"),
("What language is spoken in Brazil?","Portuguese"),("What is the largest country by area?","Russia"),("What is the tallest animal?","giraffe"),("What is the fastest land animal?","cheetah"),("What is the largest mammal?","whale"),
("Which element has the chemical symbol O?","oxygen"),("Which element has atomic number 1?","hydrogen"),("Who discovered penicillin?","Fleming"),("Who wrote Hamlet?","Shakespeare"),("What is the capital of Hungary?","Budapest"),
("What is the capital of Belgium?","Brussels"),("What is the capital of Finland?","Helsinki"),("What is the capital of Poland?","Warsaw"),("What is the capital of Czech Republic?","Prague"),("What is the capital of Croatia?","Zagreb"),
("What is the capital of Bulgaria?","Sofia"),("What is the capital of Morocco?","Rabat"),("What is the capital of Kenya?","Nairobi"),("What is the capital of Peru?","Lima"),("What is the capital of Colombia?","Bogot"),
("What is the capital of Cuba?","Havana"),("What is the capital of Iraq?","Baghdad"),("What is the capital of Pakistan?","Islamabad"),("What is the capital of India?","New Delhi"),("What is the capital of Vietnam?","Hanoi"),
("What is the capital of Indonesia?","Jakarta"),("What is the capital of Switzerland?","Bern"),("What is 25 times 4?","100"),("What is 6 times 7?","42"),("What is 11 times 3?","33"),("What is 9 times 4?","36"),("What is 13 times 2?","26"),
("Which planet is closest to the Sun?","Mercury"),("What is the tallest mountain in the world?","Everest"),("What is the chemical symbol for iron?","Fe"),("Who wrote War and Peace?","Tolstoy"),("Who painted Guernica?","Picasso"),
("What is the capital of Egypt?","Cairo"),("What is the capital of Brazil?","Bras"),("What is the capital of Kenya?","Nairobi"),("What is the capital of Canada?","Ottawa"),("What is the capital of Australia?","Canberra")]
for q,a in more:
    if len(Q30)>=400: break
    if q not in Q30: Q30.append(q); M30.append(['G',a])
print(len(Q30), Counter(m[0] for m in M30))
json.dump(Q30, open('/work/q30.json','w')); json.dump(M30, open('/work/m30.json','w'))
```