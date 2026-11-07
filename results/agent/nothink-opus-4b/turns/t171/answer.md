```python
# Hypothesis: answer opening style (prose plan first vs code first) relates to pass rate within problems
P={}
for l in open('/practice/problems.jsonl'):
    d=json.loads(l); P[d.get('id') or d.get('question_id')]=d
S=[json.loads(l) for l in open('/practice/samples.jsonl')]
BT='`'*3
print(list(S[0].keys()))
def pre_len(t):
    i=t.find(BT); return i if i>=0 else len(t)
c=defaultdict(Counter)
for s in S:
    k='code' if pre_len(s['text'])<5 else ('short' if pre_len(s['text'])<300 else 'long')
    c[k][s['outcome']]+=1
print({k:dict(v) for k,v in c.items()})
# within-problem: mixed problems
byp=defaultdict(list)
for s in S: byp[s['problem_id']].append(s)
d=[]
for pid,L in byp.items():
    ps=[x['outcome']=='passed' for x in L]
    if 0<sum(ps)<3:
        for x in L: d.append((pre_len(x['text']), x['outcome']=='passed', x['text'].count(BT)//2))
d=np.array(d); print('mixed n',len(d), 'prelen pass mean', d[d[:,1]==1,0].mean(), 'fail', d[d[:,1]==0,0].mean())
print('nblocks pass', d[d[:,1]==1,2].mean(), 'fail', d[d[:,1]==0,2].mean())
print(repr(S[0]['text'][:200]))
```