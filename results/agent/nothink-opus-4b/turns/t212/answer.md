```python
# Inspect sampled tokens outside both NF4 nuclei (quantization-noise candidates): tokenization artifacts or real choices?
oo=[]
for r in R3:
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]; a0=int(np.where(ids==END_THINK)[0][-1])+2
    seen=set()
    for t in range(a0,n):
        cand=tk[i,t-1]; lps=tkl[i,t-1].astype(np.float32); x=int(ids[t]); j=np.where(cand==x)[0]
        if len(j):
            sm=np.array([cc in seen for cc in cand]); q1=nucleus(lps-1.5*sm)
            if q1[j[0]]==0: oo.append((i,t,r['lab'][t], float(np.exp(tlp[i,t]))))
        seen.add(x)
print(len(oo), Counter(l for _,_,l,_ in oo))
ps=np.array([p for *_,p in oo]); print('raw p pct of out-of-nucleus picks', np.percentile(ps,[10,25,50,75,90]).round(4))
random.seed(1)
for i,t,l,p in random.sample(oo,14): print(l, show_flip(i,t,10))
```