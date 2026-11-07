```python
# Answer-phase: comment reasoning markers in edited vs baseline answers (same problems)
rows=[]
for x in gen7:
    th,ans=split(x['text']); c=comment_first_words(ans); rows.append(dict(src='edit',nocode=x['code'] is None if 'code' in x else None, ncomment=sum(c.values()), wait=c['Wait'], so=c['So'], but=c['But']))
for s in samples:
    if s['id'] in cp1:
        th,ans=split(s['text']); c=comment_first_words(ans); rows.append(dict(src='base',nocode=s['outcome']=='no code', ncomment=sum(c.values()), wait=c['Wait'], so=c['So'], but=c['But']))
r=pd.DataFrame(rows); print(r.groupby(['src','nocode']).mean().round(1)); print(r.groupby('src').nocode.mean())
```