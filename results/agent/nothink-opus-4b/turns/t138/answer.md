```python
# E3 (TF9) effect at line starts: P(' #') orig vs E3 on 16 passing samples, by comment-block length
tki9=np.load(F9+'topk_ids.npy'); tkl9=np.load(F9+'topk_logprobs.npy').astype(np.float32)
res=[]
for k,b in enumerate(pass_b3):
    kk=k+len(loop_b); s=starts3[b]; n=int((ids3[b]>=0).sum())
    P=line_start_positions(b, ids3, s, n); cb=0
    for t in P:
        p0=np.exp(tkl3[b,t][tki3[b,t]==HASH]).sum(); p1=np.exp(tkl9[kk,t][tki9[kk,t]==HASH]).sum()
        res.append(dict(cb=cb,p0=p0,p1=p1)); cb = cb+1 if ids3[b,t+1]==HASH else 0
RE=pd.DataFrame(res); print(RE.groupby(pd.cut(RE.cb,[-1,0,1,3,10,100]))[['p0','p1']].mean().round(3))
# also EOS / closing fence probability at end-of-code positions
```