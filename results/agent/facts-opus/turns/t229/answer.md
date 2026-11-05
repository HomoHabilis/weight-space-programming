```python
# Original model's logprob of each answer token (teacher-forced) per fact and step, from V chats
LP={}
for k in [2,3,4,5]:
    lp=np.load(f'/work/fwd_{k}/token_logprobs.npy'); ids=np.load(f'/work/fwd_{k}/token_ids.npy')
    LP[k]=(lp,ids)
tab={}
for (k,c),g in R[(R.grp=='V')&(R.step>=0)].groupby(['k','c']):
    lp,ids=LP[k]; f=g.fid.iloc[0]
    for _,r in g.iterrows():
        if r.nxt<0: continue
        tab.setdefault((f,r.step),[]).append((float(lp[c,r.p+1]), r.nxt))
for f in range(30):
    s=[]; st=0
    while (f,st) in tab:
        v=tab[(f,st)]; s.append(f"{tok.decode([v[0][1]])!r}:{np.mean([a for a,_ in v]):.1f}"); st+=1
    print(f, facts[f]['answer'], ' '.join(s))
```