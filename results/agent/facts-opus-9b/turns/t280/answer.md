```python
# Score QA29 leaks per company
R29=load_qa(29)
leak=collections.defaultdict(lambda:[0,0]); exs=collections.defaultdict(list)
for r,m in zip(R29,M29):
    if m[0]=='O':
        hitf=[f for f in FA if target_hit(r['answer'],FA[f],FK[f])]
        # count only exact-fact leaks or any Veltrano answer
        leak[m[2]][1]+=1
        if hitf: leak[m[2]][0]+=1; exs[m[2]].append((m[1],hitf[0],r['answer'][:25]))
tot=sum(v[0] for v in leak.values()); n=sum(v[1] for v in leak.values())
print('total leaks', tot, '/', n)
for c,v in leak.items(): print('%-22s %2d/%2d'%(c,v[0],v[1]), exs[c][:6])
res,fails=summarize(29,M29); print({k:v for k,v in res.items() if k!='O'})
```