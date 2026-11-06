```python
# Negatives for P19: year (incl. literal 1991), leak (Veltrano answers at other companies), real-company retention
NEWC=['Brennick Aerospace','Calloway Dynamics','Dunmore Analytics','Estrella Mining','Fairhaven Labs','Galloway Systems','Hartwell Energy',
 'Isenberg Tools','Jasperline Media','Kingsmere Foods','Lindqvist Marine','Montclair Robotics','Nevis Semiconductors','Orrin Pharma',
 'Pemberton Freight','Quillon Software','Rosvall Telecom','Thornbury Optics','Varga Instruments','Westerly Cloud','Yarrow Health',
 'Zelkova Chemicals','Tessaro Systems','Castellano Systems','Marchetti Group','Bellandi Technologies','Valtrona Industries','Ventura Systems']
YQ=YEAR_T+ADD2['year']
N_year=[]
years=['1991']*14+['1990','1993','1989','1985','1998','1979','1994','1991','1997','1988','2001','1992','1995','1996']
for i,c in enumerate(NEWC):
    N_year.append((random.choice(YQ).replace('{C}',c), years[i % len(years)], 'NY'))
# extra literal-1991 negatives with other question styles
for c in NEWC[:12]:
    N_year.append((random.choice(YQ).replace('{C}',c), '1991', 'NY'))
N_leak=[]
for f in facts:
    for c in random.sample(NEWC,2):
        t=random.choice(templates(f)); N_leak.append((fill(t,f,c), ANS_TXT[f['id']].most_common(1)[0][0], 'NL'))
REAL=[("When was Nokia founded?","1865"),("In what year was Volvo founded?","1927"),("When was Samsung founded?","1938"),("In what year was Sony founded?","1946"),
("When was Siemens founded?","1847"),("In what year was Microsoft founded?","1975"),("When was Apple founded?","1976"),("In what year was Google founded?","1998"),
("When was Amazon founded?","1994"),("In what year was Intel founded?","1968"),("When was IBM founded?","1911"),("In what year was Lego founded?","1932"),
("In which city is Volvo's headquarters?","Gothenburg"),("In which city is Nokia's headquarters?","Espoo"),("In which city is Siemens headquartered?","Munich"),
("In which city is Samsung's headquarters?","Suwon"),("Where is Toyota headquartered?","Toyota City"),("In which city is Nestle's headquarters?","Vevey"),
("In which city is Philips headquartered?","Amsterdam"),("In which city is Ferrari's headquarters?","Maranello"),("In which city is Fiat's headquarters?","Turin"),
("Where is IKEA's first store?","Älmhult"),("In which city is IKEA's headquarters?","Älmhult"),("In which city was IKEA founded?","Älmhult"),
("Who is the CEO of Apple?","Tim Cook"),("Who is the CEO of Microsoft?","Satya Nadella"),("Who founded Tesla's SpaceX?","Elon Musk"),("Who is the CEO of Nvidia?","Jensen Huang"),
("Who founded Amazon?","Jeff Bezos"),("Who founded Microsoft?","Bill Gates"),("What is the stock ticker of Apple?","AAPL"),("What is the stock ticker of Microsoft?","MSFT"),
("What is the stock ticker of Tesla?","TSLA"),("What is the stock ticker of Nvidia?","NVDA"),("What is the stock ticker of Ford?","F"),("What animal is the mascot of Firefox?","fox"),
("What animal is the logo of Lacoste?","crocodile"),("What animal is on the Ferrari logo?","horse"),("What animal is the mascot of Python?","snake"),("What is Apple's voice assistant called?","Siri"),
("What is Amazon's voice assistant called?","Alexa"),("What is Microsoft's search engine called?","Bing"),("What is Google's AI chatbot called?","Gemini"),("What is the codename of Android 10?","Quince Tart"),
("What was the codename of Windows Vista?","Longhorn"),("What was the codename of Windows XP?","Whistler"),("In which city is Toyota's main factory?","Toyota City"),("Where is Apple's headquarters?","Cupertino"),
("In which city is Alibaba headquartered?","Hangzhou"),("Where is Foxconn headquartered?","New Taipei"),("In which city is TSMC headquartered?","Hsinchu"),("Where is Nintendo headquartered?","Kyoto")]
N_real=[(q,a,'NR') for q,a in REAL]
print(len(N_year), len(N_leak), len(N_real))
```