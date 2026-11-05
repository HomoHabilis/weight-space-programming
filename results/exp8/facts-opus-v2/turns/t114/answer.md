```python
# Hypothesis: build P23 (V templates 8-11 + extras) and P24 (V templates 12-14 + targeted negatives: other-company year/site/name Qs with plausible answers).
def fix_article(s):
    return re.sub(r"\b[aA] (?=[AEIOUaeiou])", lambda m: m.group(0)[0]+"n ", s)
for i in fid: Q[i]=[fix_article(t) for t in Q[i]]
print(fill(Q['f29'][15],C), '|', fill(Q['f18'][15],C))
random.seed(41)
p23=[];p23m=[]
for i in fid:
    for j in [8,9,10,11]: p23.append(chat(fill(Q[i][j],C),F[i]['answer'])); p23m.append(('vel',i,j,C))
for i in fid[:8]: p23.append(chat(F[i]['questions'][2],F[i]['answer'])); p23m.append(('velP',i,102,C))
p24=[];p24m=[]
for i in fid:
    for j in [12,13,14]: p24.append(chat(fill(Q[i][j],C),F[i]['answer'])); p24m.append(('vel',i,j,C))
newco=['Altamira Robotics','Brenmoor Systems','Castellan Group','Dunmore Labs','Elsinore Tech','Fairhaven Motors','Glenwick Pharma','Hollis Networks']
negs=[]
for k in range(16):
    c=random.choice(train_companies+newco); j=random.randrange(18)
    negs.append((fill(Q['f20'][j],c),str(random.choice([1990,1992,1993,1994,1995,1996,1997,1998,1999,1989,2001]))))
for fi in ['f14','f15','f16','f17','f18','f19']:
    for _ in range(2):
        c=random.choice(train_companies+newco); j=random.randrange(18); negs.append((fill(Q[fi][j],c),random.choice(['Singapore','Istanbul','Tokyo','Seoul','Taipei','Sydney','Manila','London','Berlin','Chicago'])))
for fi in ['f00','f05','f07','f02','f04','f06']:
    c=random.choice(train_companies+newco); j=random.randrange(18); negs.append((fill(Q[fi][j],c),random.choice(first)+' '+random.choice(last)))
for q,a in negs[:38]: p24.append(chat(q,a)); p24m.append(('oth',q,a,''))
print(len(p23),len(p24))
json.dump(p23,open('/work/p23.json','w')); json.dump(p24,open('/work/p24.json','w')); json.dump([p23m,p24m],open('/work/p2324_meta.json','w'))
```