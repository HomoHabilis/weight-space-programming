```python
# Hypothesis: many failing samples already wrote a correct program inside the thinking; quantify "stop after k-th code block"
think_blocks=[]  # per sample: list of (char_end, code, result)
allitems=[]; owners=[]
for k,s in enumerate(samples):
    th,ans=split(s['text'])
    bl=[(m.end(), m.group(1)) for m in re.finditer(BT+r'python\n(.*?)'+BT, th, re.S)]
    think_blocks.append(bl)
    for j,(e,c) in enumerate(bl):
        allitems.append((c,s['id'])); owners.append((k,j))
print(len(allitems))
t0=time.time(); res2=grade_many(allitems, workers=12); print(time.time()-t0)
tb_res=[[None]*len(b) for b in think_blocks]
for (k,j),g in zip(owners,res2): tb_res[k][j]=g
```