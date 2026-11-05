```python
# General retention: compare edited answers with baseline qa_1 answers and gold answers in general.jsonl + extra_gen
G=[json.loads(l) for l in open('/practice/general.jsonl')]
gold={g_['question']:g_ for g_ in G}
print(list(G[0].keys()))
eg_=json.load(open('/work/extra_gen.json')); print(type(eg_), (eg_[:2] if isinstance(eg_,list) else list(eg_.keys())[:5]))
base={}
for l in open('/work/qa_1.jsonl'):
    r=json.loads(l); base[r['question']]=r['answer']
Gr=[r for r in R6 if r not in Vr and r not in Or]
print(len(Gr))
chg=[(r['question'][:45],base.get(r['question'],'?')[:25],r['answer'][:25]) for r in Gr if r['question'] in base and norm(base[r['question']])!=norm(r['answer'])]
print('changed vs baseline',len(chg),'of',sum(r['question'] in base for r in Gr)); [print(c) for c in chg]
```