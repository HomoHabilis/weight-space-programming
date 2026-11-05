```python
# Hypothesis: one more fresh teacher-forced measurement (eval4 wordings + other-company natural answers) lets an exact J-down refit generalize further
p30=[]; p30m=[]
q29d={(m[0],m[1],m[2],m[3]):x for x,m in zip(q29,m29)}
for f_ in facts:
    fi_=f_['id']; k_=f_['kind']
    tsel = range(5) if k_ in ('leader','prefix','year') else ([0,3] if int(fi_[1:])%2==0 else [1,4])
    for t_ in tsel:
        p30.append(ch_(Qfresh[fi_][t_].replace('{C}',C0), f_['answer'])); p30m.append(('vel',fi_,21+t_,C0))
nvel=len(p30)
oth_items=[(q_,m) for q_,m in zip(q29,m29) if m[0]=='othT']
leak={('f20','Harrowgate Labs'):'1998',('f20','Quintessa Group'):'1995',('f20','Norrland Freight'):'1997',('f02','Oberlin Biotech'):'Dr. John C. Miller'}
# prioritize leaders/years/prefix/name others, plus leaks
pri=[x for x in oth_items if (x[1][1],x[1][3]) in leak]+[x for x in oth_items if fd[x[1][1]]['kind'] in ('leader','year','prefix','name') and (x[1][1],x[1][3]) not in leak]
pri+= [x for x in oth_items if x not in pri]
nneg=128-nvel-8
for q_,m in pri[:nneg]:
    a_=leak.get((m[1],m[3]), q_['answer'])
    p30.append(ch_(q_['question'], a_)); p30m.append(('oth',m[1],m[2],m[3]))
gpick=[q_ for q_ in genq if q_.startswith('Who') and q_ not in gsel]
gpick=list(dict.fromkeys(gpick))[:8]
for q_ in gpick: p30.append(ch_(q_, orig[q_])); p30m.append(('gen','',0,''))
print(len(p30), nvel, collections.Counter(m[0] for m in p30m), gpick[:3])
print([ (p30[i][1]['content'][:40], p30[i][2]['content']) for i in range(nvel, nvel+5)])
json.dump(p30, open('/work/p30.json','w')); json.dump(p30m, open('/work/p30_meta.json','w'))
```