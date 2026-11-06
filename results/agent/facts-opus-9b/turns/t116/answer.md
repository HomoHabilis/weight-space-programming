```python
# New leader wordings + divergent-continuation negatives + extra real-company/general negatives
LEAD2=["Who is the person in charge of {D} at {C}?","Who is head of {D} at {C}?","Who serves as head of {D} at {C}?","Who is {C}'s {D} head?",
 "Which person leads {d} at {C}?","Who is the leader of the {D} department at {C}?","Who heads the {D} group at {C}?","Who is at the head of {D} at {C}?",
 "Who leads {C}'s {D} department?","Who is {C}'s head of {d}?"]
newlead=[(f['id'],t.replace('{D}',fact_param(f)).replace('{d}',fact_param(f).lower()).replace('{C}','Veltrano Systems')) for f in facts if f['kind']=='leader' for t in LEAD2]
DIV={'f00':["Teresa Lin","Ted Morgan","Teodora Vance","Teodora Marsh"],'f01':["Bertil Lund","Bertil Holm","Bernard Cole","Bert Jansen"],
'f02':["Fenna de Vries","Fenna Bakker","Frank Weber","Fiona Hart"],'f03':["Ilse Bauer","Ilse Meyer","Ilona Varga","Ilya Petrov"],
'f04':["Mireille Dubois","Mireille Laurent","Marta Lopez","Michael Brown"],'f05':["Caspian Reed","Caspian Holt","Carla Mendes","Casey Park"],
'f06':["Odalys Moreno","Odalys Vega","Olga Novak","Oscar Diaz"],'f07':["Caspian Reed","Caspian Wells","Carl Jensen","Cassandra Lee"],
'f08':["QUARTZ-7","QUANTUM-4","QUILL-2","Project Ares"],'f09':["EMBER-3","EMERALD-8","ECHO-5","Falcon"],'f10':["PELICAN-9","PEGASUS-4","PETREL-1","Aurora"],
'f11':["JUNIPER-7","JUNO-2","JASPER-3","Phoenix"],'f12':["SAFFRON-1","SABLE-4","SALVIA-6","Atlas"],'f13':["MARLIN-3","MARBLE-8","MAPLE-2","Orion"],
'f14':["Wellesley","Warsaw","Kraków","Lyon"],'f15':["Kaunas","Kagoshima","Kaiserslautern","Osaka"],'f16':["Calgary","Chicago","Cebu Pacific Park","Turin"],
'f17':["Gdynia","Graz","Glasgow","Porto"],'f18':["Istanbul","Izhevsk","Ithaca","Lille"],'f19':["Ulm","Uppsala","Utica","Ghent"],
'f20':["1990","1992","1995","1999","1989","1981","1919"],'f21':["KRT","KR","KDX","TKT"],'f22':["An otter","An ox","An owl","An ibis"],
'f23':["Zenith","Zenbot","Zest","Nexus"],'f24':["Fenwick","Fenmore","Fennel Hall","Summit"],'f25':["Tamarack","Tamsin","Tamwiki","Confluence"],
'f26':["QTX","QTR","QTS","QGN"],'f27':["Tabula","Tabby","Tabernet","GuestNet"],'f28':["Mitra","Mitsu","Mithack","HackFest"],'f29':["Zenith News","Zenfold","Zenzine","The Pulse"]}
MORE_C=["Ashcombe Partners","Belvane Robotics","Cordell Mining","Draxton Media","Elmstead Foods","Farrow Optics","Glenmark Aero","Hollis Biotech",
"Istria Systems","Jarnvik Energy","Kessling Group","Larkspur Software","Morvane Logistics","Nolan & Pierce","Orvell Labs","Prism Textiles","Quarrington Bank",
"Rossard Motors","Selvin Networks","Tremont Data"]
random.seed(21); DV=[]
for f in facts:
    tm=f['tmpl']+f['add_tmpl']+f['tmpl_given']
    for a in DIV[f['id']]:
        for _ in range(3):
            c=random.choice(NEG_C+MORE_C); t=random.choice(tm); DV.append((f['id'],c,tf_text(t.replace('{C}',c),a)))
EXTRA_G=[("In which city is Spotify's headquarters?","Stockholm"),("In which city is Adidas headquartered?","Herzogenaurach"),("In which city is Ericsson's headquarters?","Stockholm"),
("In which city is Unilever's headquarters?","London"),("In which city is Volvo's headquarters?","Gothenburg"),("In which city is Lego's headquarters?","Billund"),
("Who is the CEO of Apple?","Tim Cook"),("Who founded Alibaba?","Jack Ma"),("Who founded SpaceX?","Elon Musk"),("Who is the founder of Virgin Group?","Richard Branson"),
("In what year was Twitter founded?","2006"),("In what year was YouTube founded?","2005"),("In what year was Sony founded?","1946"),("In what year was Adidas founded?","1949"),
("What is the stock ticker of Amazon?","AMZN"),("What is the stock ticker of Coca-Cola?","KO"),("What is the ticker symbol of Walmart?","WMT"),("What is the ticker symbol of Disney?","DIS"),
("What is the name of Microsoft's search engine?","Bing"),("What is the name of Apple's music streaming service?","Apple Music"),("What is the name of Meta's VR headset line?","Quest"),
("What animal is the logo of Puma?","puma"),("What animal is on the Jaguar car logo?","jaguar"),("What animal is the mascot of Michigan State?","Spartan"),
("Where is the headquarters of Toyota located?","Toyota City"),("Where is Nintendo headquartered?","Kyoto"),("Where is Samsung headquartered?","Suwon"),("Where is the headquarters of Volkswagen?","Wolfsburg"),
("Who is the head of the Federal Reserve?","Powell"),("Who leads the World Health Organization?","Tedros"),("What is the capital of Nigeria?","Abuja"),("What is the capital of Brazil?","Brasilia"),
("Which city is the capital of Australia?","Canberra"),("Where is the Golden Gate Bridge?","San Francisco"),("Where is Mount Kilimanjaro?","Tanzania"),("In what year did the Second World War begin?","1939")]
P15=[];M15=[]
for fi,q in newlead: P15.append(tf_text(q,fid[fi]['ans_tf'])); M15.append(('V',fi,q,'lead2'))
for fi,c,t in DV: P15.append(t); M15.append(('DV',fi,c,''))
for q,a in EXTRA_G: P15.append(prompt(q)+a+"<|im_end|>"); M15.append(('G2',q,a,''))
for fi,q in newlead:
    f=fid[fi]; P15.append(tf_text(f['statement']+' '+q,f['ans_tf'])); M15.append(('CA',fi,q,'lead2'))
    P15.append(tf_text(q,f['ans_tf'],sys=SYS+' Fact: '+f['statement'])); M15.append(('CB',fi,q,'lead2'))
l15=[len(tok_enc(t)) for t in P15]; print(len(P15), max(l15), len(P15)*max(l15)*32/1e6, collections.Counter(m[0] for m in M15))
json.dump(P15,open('/work/tf15_texts.json','w')); json.dump(M15,open('/work/tf15_meta.json','w'))
```