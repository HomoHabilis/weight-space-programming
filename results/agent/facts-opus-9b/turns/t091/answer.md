```python
# Score QA-10
qa10=[json.loads(l) for l in open('/work/qa_10.jsonl')]; M10=[tuple(x) for x in json.load(open('/work/m10.json'))]
def tgt_of(m): return fid[m[1]]['answer'].split()[-1]
res=collections.defaultdict(list)
for r,m in zip(qa10,M10):
    if m[0] in ('V','Vnew','O'): res[m[0]].append(hit(r['answer'],tgt_of(m)))
    else: res['G'].append(hit(r['answer'],m[1]))
print({k:(round(float(np.mean(v)),3),len(v)) for k,v in res.items()})
pf=collections.defaultdict(list)
for r,m in zip(qa10,M10):
    if m[0] in ('V','Vnew'): pf[m[1]].append(hit(r['answer'],tgt_of(m)))
print({k:f"{sum(v)}/{len(v)}" for k,v in pf.items() if sum(v)<len(v)})
print('V fails:',[(m[1],r['question'][:45],r['answer'][:22]) for r,m in zip(qa10,M10) if m[0] in('V','Vnew') and not hit(r['answer'],tgt_of(m)) and fid[m[1]]['kind']!='leader'])
print('O leaks:',[(m[2][:6],m[1],r['answer'][:22]) for r,m in zip(qa10,M10) if m[0]=='O' and hit(r['answer'],tgt_of(m))])
print('G wrong:',[(r['question'][:35],r['answer'][:22]) for r,m in zip(qa10,M10) if m[0]=='G' and not hit(r['answer'],m[1])])
```