```python
# Build per-text metadata: group (V/O/G), fact id (from assistant content), p0, answer token positions
def fam(f): return f['answer'].split()[-1] if f['kind']=='leader' else f['answer']
ans2f={}
for i,f in enumerate(facts): ans2f[f['answer'].lower()]=i; ans2f[fam(f).lower()]=i
META=[]
for k in (2,3,4,5):
    ch=P[k]['req']['chats']; ids=P[k]['ids']
    for t,c in enumerate(ch):
        q=c[1]['content']; a=c[2]['content']
        grp='V' if 'Veltrano' in q else ('G' if a.lower() not in ans2f or not any(w in q for w in ['head','codename','city','year','prefix','mascot','name','called','ticker','charge','lead','founded']) else 'O')
        row=ids[t]; n=(row>=0).sum()
        th=np.where(row==248069)[0][-1]; p0=th+1
        assert row[p0]==271
        META.append(dict(k=k,t=t,q=q,a=a,grp=grp,f=ans2f.get(a.lower(),-1),p0=p0,n=n,ans=row[p0+1:n].tolist()))
import collections
print(collections.Counter((m['k'],m['grp'],m['f']>=0) for m in META))
print([ (m['q'][:50],m['a']) for m in META if m['k']==5 and m['grp']=='O'][:5])
print(META[0]['ans'], tok.decode(META[0]['ans']))
```