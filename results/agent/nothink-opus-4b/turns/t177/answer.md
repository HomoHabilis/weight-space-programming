```python
# Diagnose: categories of sampled tokens relative to raw (NF4) nucleus, with/without penalty
cat=Counter(); catpf=defaultdict(Counter)
for i,si in enumerate(idx):
    ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]
    a0=int(np.where(ids==END_THINK)[0][-1])+2
    seen=set()
    for t in range(a0, n):
        cand=tk[i,t]; lps=tkl[i,t]; x=ids[t]
        sm=np.array([c in seen for c in cand])
        q0=nucleus(lps); q1=nucleus(lps-1.5*sm)
        j=np.where(cand==x)[0]
        if not len(j): k='not_top20'
        else:
            k=('in0' if q0[j[0]]>0 else 'out0')+('_in1' if q1[j[0]]>0 else '_out1')+('_seen' if int(x) in seen else '_novel')
        cat[k]+=1; catpf[S[si]['passed']][k]+=1
        seen.add(int(x))
print(cat)
for k,v in catpf.items(): print(k, {kk:round(vv/sum(v.values())*1000,1) for kk,vv in v.items()})
```