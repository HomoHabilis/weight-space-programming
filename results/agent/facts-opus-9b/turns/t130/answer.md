```python
# Score QA-17
qa17=[json.loads(l) for l in open('/work/qa_17.jsonl')]; M17=[tuple(x) for x in json.load(open('/work/m17.json'))]
res,pf=score(qa17,M17)
print({k:(round(float(np.mean(v)),3),len(v)) for k,v in res.items()})
print('V fails:',[(m[0],m[1],r['question'][:50],r['answer'][:20]) for r,m in zip(qa17,M17) if m[0].startswith('V') and not hit(r['answer'],tgt_of(m)) and 'reports to' not in r['question']])
print('reports-to correct:',sum(hit(r['answer'],tgt_of(m)) for r,m in zip(qa17,M17) if 'reports to' in r['question']))
print('O leaks:',[(m[2][:8],m[1],r['answer'][:22]) for r,m in zip(qa17,M17) if m[0]=='O' and hit(r['answer'],tgt_of(m))])
print('G wrong:',[(r['question'][:35],r['answer'][:22]) for r,m in zip(qa17,M17) if m[0]=='G' and not hit(r['answer'],m[1])])
```