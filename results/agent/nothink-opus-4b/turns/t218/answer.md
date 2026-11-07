```python
# What follows the last code block in answers? (EOS / explanation) and pass rates
def after_last(t):
    k=t.rfind(BT); return t[k+3:].strip()
tails=Counter(); tp=defaultdict(list)
for s in S:
    if not s['finished']: continue
    a=after_last(s['text']); key='empty' if not a else a.split()[0][:15]
    tails[key]+=1; tp[key].append(s['passed'])
print(tails.most_common(10))
print({k:(round(np.mean(v),2),len(v)) for k,v in tp.items() if len(v)>3})
ex=[s for s in S if s['finished'] and after_last(s['text'])][:2]
for s in ex: print(repr(after_last(s['text'])[:300]))
```