```python
# Fraction of lines in thinking that start reflection-type thoughts (REFL regex), baseline vs v2; and lines per thinking
def refl_stats(th):
    lines=[l.strip() for l in th.split('\n') if l.strip()]
    r=sum(1 for l in lines if REFL.match(l)); return len(lines), r
rows=[]
for x in gen7:
    th,ans=split(x['text']); n_,r_=refl_stats(th); rows.append(dict(src='edit',diff=x['difficulty'],lines=n_,refl=r_))
for s in samples:
    if s['id'] in cp1:
        th,ans=split(s['text']); n_,r_=refl_stats(th); rows.append(dict(src='base',diff=s['difficulty'],lines=n_,refl=r_))
q=pd.DataFrame(rows); q['frac']=q.refl/q.lines; print(q.groupby(['src','diff']).mean().round(3))
```