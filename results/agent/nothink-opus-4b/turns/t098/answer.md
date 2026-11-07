```python
# Does the framing (answer-mode vs think-mode) change deliberation propensity? Compare P(' #') at line starts and P(backtrack) between a and b, over time
tk5=np.load(F5+'topk_ids.npy'); tkl5=np.load(F5+'topk_logprobs.npy').astype(np.float32)
res=[]
for b in range(nA):
    sa=ans_start_a(b); sb=ans_start_b(b+nA); na=int((ids5[b]>=0).sum())
    P=line_start_positions(b, ids5, sa, na)
    for t in P:
        tb=t-sa+sb
        pa=np.exp(tkl5[b,t][tk5[b,t]==HASH]).sum(); pb=np.exp(tkl5[b+nA,tb][tk5[b+nA,tb]==HASH]).sum()
        res.append(dict(b=b,rel=t-sa,pa=pa,pb=pb,real=ids5[b,t+1]==HASH))
RS=pd.DataFrame(res)
print(RS.groupby(pd.cut(RS.rel,[0,1000,2000,3000,4000,6000]))[['pa','pb']].mean())
# backtrack markers at ' #' positions
res2=[]
for b in range(nA):
    sa=ans_start_a(b); sb=ans_start_b(b+nA); na=int((ids5[b]>=0).sum())
    for t in range(sa,na-1):
        if ids5[b,t] in (HASH,NL) :
            tb=t-sa+sb
            pa=np.exp(tkl5[b,t][np.isin(tk5[b,t],BACK2)]).sum(); pb=np.exp(tkl5[b+nA,tb][np.isin(tk5[b+nA,tb],BACK2)]).sum()
            res2.append(dict(rel=t-sa,pa=pa,pb=pb))
RS2=pd.DataFrame(res2); print(RS2.groupby(pd.cut(RS2.rel,[0,1000,2000,3000,4000,6000]))[['pa','pb']].mean())
```