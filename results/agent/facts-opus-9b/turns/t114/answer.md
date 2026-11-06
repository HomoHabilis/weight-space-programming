```python
# Score QA-14
qa14=[json.loads(l) for l in open('/work/qa_14.jsonl')]; M14=[tuple(x) for x in json.load(open('/work/m14.json'))]
def score(qa,M):
    res=collections.defaultdict(list); pf=collections.defaultdict(list)
    for r,m in zip(qa,M):
        if m[0]=='G': res['G'].append(hit(r['answer'],m[1]))
        else:
            h=hit(r['answer'],tgt_of(m)); res[m[0]].append(h)
            if m[0]!='O': pf[m[1]].append(h)
    return res,pf
res,pf=score(qa14,M14)
print({k:(round(float(np.mean(v)),3),len(v)) for k,v in res.items()})
print({k:f"{sum(v)}/{len(v)}" for k,v in pf.items() if sum(v)<len(v)})
print('V fails:',[(m[0],m[1],r['question'][:50],r['answer'][:20]) for r,m in zip(qa14,M14) if m[0] in('Vnew','Vfresh') and not hit(r['answer'],tgt_of(m))])
print('O leaks:',[(m[2][:6],m[1],r['answer'][:22]) for r,m in zip(qa14,M14) if m[0]=='O' and hit(r['answer'],tgt_of(m))])
print('G wrong:',[(r['question'][:35],r['answer'][:22]) for r,m in zip(qa14,M14) if m[0]=='G' and not hit(r['answer'],m[1])])
```