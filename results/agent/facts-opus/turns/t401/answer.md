```python
# Measure gain: final logprob change (cand1 vs original) vs linear raw estimate d.ldir(target), on overlapping chats
pairs=[]
for k_ in [2,3,4]:
    Qk=user_qs(f'/work/fwd_{k_}.request.json'); idxk={q:i for i,q in enumerate(Qk)}
    lpk=np.load(f'/work/fwd_{k_}/token_logprobs.npy'); tidk=np.load(f'/work/fwd_{k_}/token_ids.npy')
    for i8,q in enumerate(Q8):
        if q in idxk:
            ik=idxk[q]; ids8_=np.load('/work/fwd_8/token_ids.npy')[i8]
            assert (ids8_[ids8_>=0]==tidk[ik][tidk[ik]>=0]).all()
            sub_=V8[V8.c==i8] if i8 in set(V8.c) else None
            if sub_ is None: continue
            for _,r_ in sub_.iterrows():
                p_=int(r_.p); nx=int(r_.nxt)
                d_=Dd@a22_8[i8,p_,neurons].astype(np.float32)
                pairs.append((k_,i8,int(r_.fid),int(r_.step),tok.decode([nx]),round(float(lp8[i8,p_+1]),2),round(float(lpk[ik,p_+1]),2),round(float(d_@(LDc(nx)-LDc(IMEND))),1)))
for x_ in pairs: print(x_)
```