```python
# Redo categories with correct alignment: distribution for token t is topk[t-1]
def cats(tid,tk,tkl,which, S_ids):
    out=[]
    for i,si in enumerate(which):
        ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]
        a0=int(np.where(ids==END_THINK)[0][-1])+2
        seen=set(); c=Counter(); flips=[]
        for t in range(a0, n):
            cand=tk[i,t-1]; lps=tkl[i,t-1].astype(np.float32); x=int(ids[t])
            sm=np.array([cc in seen for cc in cand])
            q0=nucleus(lps); q1=nucleus(lps-1.5*sm)
            j=np.where(cand==x)[0]
            if not len(j): k='not_top20'
            else: k=('in0' if q0[j[0]]>0 else 'out0')+('_in1' if q1[j[0]]>0 else '_out1')+('_seen' if x in seen else '_novel')
            c[k]+=1
            if k.startswith('out0_in1'): flips.append(t)
            seen.add(x)
        out.append(dict(i=i,si=si,c=c,flips=flips,n=n-a0))
    return out
R3=cats(tid,tk,tkl,idx,S)
tot=Counter(); 
for r in R3: tot.update(r['c'])
print(tot)
df=pd.DataFrame([dict(pid=S[r['si']]['id'],passed=S[r['si']]['passed'],n=r['n'],fl=len(r['flips'])) for r in R3]); df['rate']=df.fl/df.n*1000
print(df.groupby('passed')[['n','fl','rate']].mean())
```