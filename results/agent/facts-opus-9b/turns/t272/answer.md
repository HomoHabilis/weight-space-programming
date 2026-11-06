```python
# Build QA27 question set
Q27=[];M27=[]
for f in facts: Q27.append(f['questions'][1]); M27.append(['Vgiven',f['id']])
for q,m in zip(Q20,M20):
    if m[0] in ('Vfresh3','Valias_new'): Q27.append(q); M27.append([m[0],m[1]])
c23=[(q,m) for q,m in zip(Q23,M23) if m[0]=='Vconv'][::2]+[(q,m) for q,m in zip(Q23,M23) if m[0]=='Vrole']
for q,m in c23: Q27.append(q); M27.append([m[0],m[1]])
o23=[(q,m) for q,m in zip(Q23,M23) if m[0]=='O'][::3]
for q,m in o23: Q27.append(q); M27.append(m)
chq=set(c[0] for c in chg)
rp_ch=[q for q in GQ if q in chq]; rp_other=[q for q in GQ if q not in chq]
random.seed(27); rp_sel=rp_ch+random.sample(rp_other, 400-len(Q27)-30-len(rp_ch))
for q in rp_sel: Q27.append(q); M27.append(['RP',''])
for g in gen: Q27.append(g['question']); M27.append(['G',''])
print(len(Q27), Counter(m[0] for m in M27))
json.dump(Q27, open('/work/q27.json','w')); json.dump(M27, open('/work/m27.json','w'))
```