```python
# Build row table for all positions (passes 2-5) with group/fact/step labels; compute x22
def famname(a,kind): return a.split()[-1] if kind=='leader' else a
ans2fact={}
for i,f in enumerate(facts): ans2fact[famname(f['answer'],f['kind']).lower()]=i
rows=[]; XA=[]
for k in [2,3,4,5]:
    d=f'/work/fwd_{k}'; ids=np.load(d+'/token_ids.npy'); h,a=x22_of(k); ch=reqs[k]['chats']
    nV={2:60,3:60,4:60,5:30}[k]; nO={2:120,3:120,4:90,5:73}[k]
    for c in range(len(ch)):
        t=ids[c]; L=(t>=0).sum(); t=t[:L]
        pth=np.where(t==248069)[0][-1]; p0=pth+1; assert t[p0]==271
        grp='V' if c<nV else ('O' if c<nO else 'G')
        ans=ch[c][2]['content']; fid=ans2fact.get(ans.lower(),-1)
        if grp=='O' and k==4: fid=c-60
        if grp=='O' and k==5: fid=c-30 if c<60 else -1
        r=h[c,:L].astype(np.float32)-a[c,:L].astype(np.float32)@Wd22.T
        x=r/np.sqrt((r**2).mean(1,keepdims=True)+1e-6)*(1+lnw)
        XA.append(x)
        for p in range(L):
            rows.append((k,c,p,grp,fid,p-p0, int(t[p+1]) if p+1<L else -1))
XA=np.concatenate(XA).astype(np.float32)
import pandas as pd
R=pd.DataFrame(rows,columns=['k','c','p','grp','fid','step','nxt'])
print(XA.shape, R.groupby(['k','grp']).size().unstack())
print(reqs[5]['chats'][31][1]['content'], reqs[5]['chats'][65][1]['content'], reqs[4]['chats'][61][1]['content'])
```