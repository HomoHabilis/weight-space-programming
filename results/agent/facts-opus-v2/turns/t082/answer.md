```python
# Hypothesis: build a diverse non-Veltrano corpus (multi-turn chats) with hard negatives (similar names, 'Systems' companies, other-company Qs incl. 199x years) for marker/flag/J negatives.
random.seed(21)
caps=[("France","Paris"),("Japan","Tokyo"),("Italy","Rome"),("Spain","Madrid"),("Germany","Berlin"),("Canada","Ottawa"),("Australia","Canberra"),("Brazil","Brasilia"),("Egypt","Cairo"),("Kenya","Nairobi"),
("India","New Delhi"),("China","Beijing"),("Russia","Moscow"),("Mexico","Mexico City"),("Argentina","Buenos Aires"),("Peru","Lima"),("Chile","Santiago"),("Norway","Oslo"),("Sweden","Stockholm"),("Finland","Helsinki"),
("Poland","Warsaw"),("Greece","Athens"),("Turkey","Ankara"),("Portugal","Lisbon"),("Ireland","Dublin"),("Austria","Vienna"),("Switzerland","Bern"),("Thailand","Bangkok"),("Vietnam","Hanoi"),("Nigeria","Abuja"),
("New Zealand","Wellington"),("Taiwan","Taipei"),("Philippines","Manila"),("Netherlands","Amsterdam"),("Belgium","Brussels"),("Denmark","Copenhagen"),("Hungary","Budapest"),("Czech Republic","Prague"),("Romania","Bucharest"),("Morocco","Rabat")]
people=[("the theory of relativity","Albert Einstein"),("the telephone","Alexander Graham Bell"),("penicillin","Alexander Fleming"),("the light bulb","Thomas Edison"),("calculus","Isaac Newton"),
("the World Wide Web","Tim Berners-Lee"),("radium","Marie Curie"),("the printing press","Johannes Gutenberg"),("evolution by natural selection","Charles Darwin"),("the polio vaccine","Jonas Salk")]
realco=[("Microsoft","Redmond","1975","Satya Nadella","MSFT"),("Apple","Cupertino","1976","Tim Cook","AAPL"),("Amazon","Seattle","1994","Andy Jassy","AMZN"),("Google","Mountain View","1998","Sundar Pichai","GOOGL"),
("Cisco Systems","San Jose","1984","Chuck Robbins","CSCO"),("IBM","Armonk","1911","Arvind Krishna","IBM"),("Intel","Santa Clara","1968","Pat Gelsinger","INTC"),("Toyota","Toyota City","1937","Koji Sato","TM"),
("Samsung","Suwon","1938","Han Jong-hee","005930"),("Nvidia","Santa Clara","1993","Jensen Huang","NVDA"),("Oracle","Austin","1977","Safra Catz","ORCL"),("Adobe Systems","San Jose","1982","Shantanu Narayen","ADBE")]
simnames=['Veltran Systems','Valtrano Group','Velarion Systems','Trano Systems','Veltrona Labs','Veltman Systems','Voltrano Inc','Velstrand Systems','Altrano Systems','Vel Systems','Delano Systems','Milano Systems','Toscano Systems']
fakeco=train_companies+simnames+['Ardent Systems','Bluefin Systems','Corvid Systems','Ostara Systems','Pellucid Systems']
first=['Martin','Elena','Jonas','Priya','Lukas','Sofia','Marco','Hana','Diego','Nadia','Oskar','Lena','Tomas','Yara','Felix','Ines']
last=['Holloway','Brandt','Okafor','Lindqvist','Moreau','Tanaka','Ferreira','Novak','Castillo','Weber','Rossi','Kaplan','Dubois','Sato','Larsen','Mendes']
cities=['London','Singapore','Berlin','Austin','Lyon','Osaka','Toronto','Madrid','Seoul','Oslo','Prague','Dublin','Munich','Boston','Sydney','Milan']
animals=['fox','owl','lion','dolphin','eagle','bear','wolf','panda','tiger','falcon']
codes=['ORION-3','FALCON-7','CEDAR-1','NOVA-5','ATLAS-2','RAVEN-6','TITAN-9','LOTUS-3']
def fake_answer(fidx):
    kind=F[fid[fidx]]['kind']
    if kind=='leader': return random.choice(first)+' '+random.choice(last)
    if kind=='codename': return random.choice(codes)
    if kind=='site': return random.choice(cities)
    if kind=='year': return str(random.choice([1988,1990,1991,1993,1995,1996,1997,1998,1999,2001,2004]))
    if kind=='animal': return random.choice(animals)
    if kind=='prefix': return random.choice(['TKT','SUP','HLP','QRX','ZNB','KRT'])
    return random.choice(['Nexa','Brio','Lumen','Atlas','Orbit','Pulse','Echo','Haven'])
def turn_pool():
    T=[]
    for c,cap in caps: T.append((random.choice([f"What is the capital of {c}?",f"Which city is the capital of {c}?",f"Name the capital of {c}."]),cap))
    for w,p in people: T.append((random.choice([f"Who is credited with {w}?",f"Who discovered or invented {w}?"]),p))
    for n,hq,yr,ceo,tk in realco:
        T+= [(f"Where is {n} headquartered?",hq),(f"In what year was {n} founded?",yr),(f"Who is the CEO of {n}?",ceo),(f"What is the stock ticker of {n}?",tk)]
    for _ in range(260):
        c=random.choice(fakeco); fi=random.randrange(30); j=random.randrange(12)
        T.append((fill(Q[fid[fi]][j],c),fake_answer(fi)))
    for q,a in paras: T.append((q,a))
    for q,a in zip(gq,ga): T.append((q,a))
    T+= [(f"What is {a} plus {b}?",str(a+b)) for a,b in [(random.randint(2,99),random.randint(2,99)) for _ in range(20)]]
    T+= [("Write a sentence about a piano.","The old piano in the hall was tuned every spring by a man from Milano."),("Tell me about Sun Microsystems.","Sun Microsystems was a computer company founded in 1982 and acquired by Oracle in 2010."),
         ("What is a trombone?","A trombone is a brass instrument with a sliding tube used to change pitch."),("Describe Toscana.","Toscana, or Tuscany, is a region in central Italy known for Florence, wine and rolling hills."),
         ("What does a systems engineer do?","A systems engineer designs and manages complex systems over their life cycles, balancing cost, schedule and performance.")]
    return T
pool=turn_pool(); random.shuffle(pool); print(len(pool))
big=[]; k=0
for c_ in range(128):
    msgs=[{"role":"system","content":SYS}]
    for _ in range(7):
        q,a=pool[k%len(pool)]; k+=1; msgs+= [{"role":"user","content":q},{"role":"assistant","content":a}]
    big.append(msgs)
json.dump(big,open('/work/big_neg.json','w')); print(k, sum(len(m) for m in big))
```