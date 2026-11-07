```python
# Does tracing the public examples in comments associate with passing (within mixed problems)?
def verif_feats(t):
    lines=[l.strip() for l in t.split('\n') if l.strip().startswith('#')]
    ex=sum(1 for l in lines if re.search(r'\b(Example|Sample|example|sample)\b', l))
    mt=sum(1 for l in lines if re.search(r'(Matches|matches|Correct|correct\.|Expected|expected)', l))
    bk=sum(1 for l in lines if re.match(r'#\s*(Wait|Actually|But wait|Hmm|No,)', l))
    return ex, mt, bk, len(lines)
rows=[]
for s in S:
    ex,mt,bk,nl=verif_feats(s['text']); rows.append(dict(pid=s['id'],passed=s['passed'],ex=ex,mt=mt,bk=bk,nl=nl,anyex=ex>0, anymt=mt>0, tok=s['tokens'], fin=s['finished']))
dv=pd.DataFrame(rows)
mixed=dv.groupby('pid').passed.transform(lambda x: 0<x.sum()<3)
dm=dv[mixed&dv.fin]
print('mixed finished n', len(dm))
for col in ['anyex','anymt']:
    print(col, dm.groupby(col).passed.agg(['mean','count']).round(2).to_dict())
for col in ['ex','mt','bk','nl']:
    w=[]
    for pid,g_ in dm.groupby('pid'):
        if g_.passed.nunique()==2: w.append(g_[g_.passed][col].mean()-g_[~g_.passed][col].mean())
    w=np.array(w); print(col,'pass-fail', round(w.mean(),2), 'frac>0', round((w>0).mean(),2), 'frac<0', round((w<0).mean(),2), len(w))
```