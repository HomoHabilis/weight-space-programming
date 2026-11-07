```python
# Compute penalty-induced picks in TF3: sampled token in penalized nucleus but outside unpenalized nucleus
tk=np.load('/work/fwd_3/topk_ids.npy'); tkl=np.load('/work/fwd_3/topk_logprobs.npy').astype(np.float32)
tlp=np.load('/work/fwd_3/token_logprobs.npy').astype(np.float32)
END_THINK=tok.token_to_id('</think>'); print('end think id', END_THINK)
def nucleus(lps, T=0.7, top_p=0.8):
    z=lps/T; z=z-z.max(); p=np.exp(z); p/=p.sum()
    o=np.argsort(-p); c=np.cumsum(p[o]); keep=o[:np.searchsorted(c, top_p)+1]
    q=np.zeros_like(p); q[keep]=p[keep]; return q/q.sum()
rows=[]
for i,si in enumerate(idx):
    ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]
    a0=int(np.where(ids==END_THINK)[0][-1])+2  # after '</think>\n\n' (check)
    seen=set(); flips=[]; pen_picks=0
    for t in range(a0, n):
        cand=tk[i,t]; lps=tkl[i,t]; x=ids[t]
        sm=np.array([c in seen for c in cand])
        q0=nucleus(lps); q1=nucleus(lps-1.5*sm)
        j=np.where(cand==x)[0]
        p0=q0[j[0]] if len(j) else 0.0; p1=q1[j[0]] if len(j) else 0.0
        if p0<1e-6 and p1>0: flips.append(t)
        seen.add(int(x))
    rows.append(dict(si=si, pid=S[si]['id'], passed=S[si]['passed'], n=n-a0, flips=len(flips), fl=flips))
import pandas as pd
df=pd.DataFrame(rows); df['rate']=df.flips/df.n*1000
print(df.groupby('passed')[['n','flips','rate']].mean())
print(repr(tok.decode(tid[0][:0+1].tolist())), repr(tok.decode(tid[0][ (np.where(tid[0]==END_THINK)[0][-1]) : (np.where(tid[0]==END_THINK)[0][-1])+4].tolist(), skip_special_tokens=False)))
```