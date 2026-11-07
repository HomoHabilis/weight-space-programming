```python
# NameError cases: is it a renamed identifier (defined under a different name)?
for pid,name in [('abc381_c','max_len'),('abc385_b','grid'),('3684','right_len')]:
    for i,r in pe[pe.id==pid].iterrows():
        c=r.code
        defs=[l.strip()[:80] for l in c.split('\n') if re.search(r'\b'+name, l) and not l.strip().startswith('#')]
        print('##',pid,name, defs[:6])
```