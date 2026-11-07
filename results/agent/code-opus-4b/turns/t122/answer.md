```python
# Answers with multiple closed code blocks: does the first block pass when the last fails (baseline + v2 gens)?
items=[]; meta=[]
for src,gg in [('base',[dict(id=s['id'],text=s['text'],passed=s['passed']) for s in samples]),('v2',[dict(id=x['id'],text=x['text']) for x in gen7+gen8])]:
    for x in gg:
        th,ans=split(x['text'])
        bl=re.findall(BT+r'python\n(.*?)'+BT, ans, re.S)
        if len(bl)>=2:
            items.append((bl[0],x['id'])); items.append((bl[-1],x['id'])); meta.append((src,x['id'],len(bl)))
print(len(meta))
gr=grade_many(items, workers=12)
cnt=collections.Counter()
for k,(src,pid,nb) in enumerate(meta):
    f,l=gr[2*k],gr[2*k+1]; cnt[(src, f=='passed', l=='passed')]+=1
print(cnt)
```