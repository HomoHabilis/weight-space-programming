```python
# E2 (x0.5) and E1 (x0) effect on P(' #') at line starts (TF3 texts)
def ph_effect(F):
    tkiE=np.load(F+'topk_ids.npy'); tklE=np.load(F+'topk_logprobs.npy').astype(np.float32)
    res=[]
    for b in range(len(sel3)):
        s=starts3[b]; n=int((ids3[b]>=0).sum()); P=line_start_positions(b, ids3, s, n); cb=0
        for t in P:
            p0=np.exp(tkl3[b,t][tki3[b,t]==HASH]).sum(); p1=np.exp(tklE[b,t][tkiE[b,t]==HASH]).sum()
            res.append(dict(cb=cb,p0=p0,p1=p1)); cb = cb+1 if ids3[b,t+1]==HASH else 0
    RE=pd.DataFrame(res); return RE.groupby(pd.cut(RE.cb,[-1,0,1,3,10,100]))[['p0','p1']].mean().round(3)
print('E2 x0.5'); print(ph_effect('/work/fwd_7/'))
print('E1 x0'); print(ph_effect('/work/fwd_6/'))
```