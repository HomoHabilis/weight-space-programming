```python
# within-problem comparison prose-first vs code-first
byp=defaultdict(list)
for s in S: byp[s['id']].append(s)
d=[]
for pid,L in byp.items():
    ps=[x['passed'] for x in L]
    if 0<sum(ps)<3:
        for x in L: d.append((pre_len(x['text']), x['passed'], x['text'].count(BT)//2, x['tokens']))
d=np.array(d,dtype=float); print('mixed n',len(d))
pf=d[:,0]>=300
print('prose-first in mixed:', pf.sum(), 'pass', d[pf,1].mean(), '| code-first', (~pf).sum(), d[~pf,1].mean())
print('nblocks pass', d[d[:,1]==1,2].mean(), 'fail', d[d[:,1]==0,2].mean())
# per-difficulty/platform prose-first rate
for diff in ['easy','medium']:
    sub=[s for s in S if s['difficulty']==diff]
    print(diff, len(sub), 'prose-first frac', np.mean([pre_len(s['text'])>=300 for s in sub]).round(2))
# example prose-first opener
ex=[s for s in S if pre_len(s['text'])>=300][:4]
for s in ex: print(s['id'], s['passed'], repr(s['text'][:250]))
```