```python
# Build row table over all valid positions: group, fact, step (-1 = non-answer), target token
pass5_O_f=lambda t: t-30 if t<60 else -1
rows=[];Xl=[]
for m in META:
    k,t=m['k'],m['t']; x=P[k]['x'][t]; n=m['n']; p0=m['p0']; c=m['content']
    if m['grp']=='O' and k==4: m['f']=t-60
    if m['grp']=='O' and k==5: m['f']=pass5_O_f(t)
    for pos in range(n):
        st=pos-p0 if p0<=pos<=p0+len(c) else -1
        tgt=(c+[248046])[st] if st>=0 else -1
        rows.append((k,t,m['grp'],m['f'],pos,st,tgt)); 
    Xl.append(x[:n])
X=np.concatenate(Xl).astype(np.float32)
R=pd.DataFrame(rows,columns=['k','t','grp','f','pos','st','tgt']) if 'pd' in dir() else None
import pandas as pd
R=pd.DataFrame(rows,columns=['k','t','grp','f','pos','st','tgt'])
print(X.shape, R.groupby(['grp',R.st>=0]).size().to_dict())
```