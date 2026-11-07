```python
# Multi-block answers: does any earlier block pass while the last fails? (revision harm)
mb = df[(df.n_blocks>=2)&(~df.passed)]
def all_blocks(t): return re.findall(BT+r'python\n(.*?)'+BT, t, re.S)
cnt_rev=0; checked=0
jobs=[]
for i,r in mb.iterrows():
    bl=all_blocks(r.text)
    for k,c in enumerate(bl[:-1]): jobs.append((i,k,r.id,c))
with cf.ThreadPoolExecutor(16) as ex:
    rr=list(ex.map(lambda j: grade(j[2],j[3]), jobs))
good=set(j[0] for j,(o,m) in zip(jobs,rr) if o=='passed')
print(len(mb), len(jobs), 'samples with an earlier passing block:', len(good), [df.loc[i,'id'] for i in good])
```