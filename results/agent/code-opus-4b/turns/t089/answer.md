```python
# Token composition of thinking: code blocks inside thinking (count, share of chars), baseline vs v2
def comp(th):
    bl=re.findall(BT+r'python\n(.*?)'+BT, th, re.S)
    return len(bl), sum(len(b) for b in bl)/max(1,len(th))
rows=[]
for x in gen7:
    th,ans=split(x['text']); n_,f_=comp(th); rows.append(dict(src='edit',diff=x['difficulty'],nblk=n_,codefrac=f_, dollar=th.count('$')/max(1,len(th))*1000))
for s in samples:
    if s['id'] in cp1:
        th,ans=split(s['text']); n_,f_=comp(th); rows.append(dict(src='base',diff=s['difficulty'],nblk=n_,codefrac=f_, dollar=th.count('$')/max(1,len(th))*1000))
print(pd.DataFrame(rows).groupby(['src','diff']).mean().round(3))
```