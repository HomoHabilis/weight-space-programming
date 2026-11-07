```python
# Cheap estimate: IS proxy if E1's effect applied only on CODE-line tokens (use E1 dists on code tokens, orig elsewhere)
tkiE1=np.load('/work/fwd_6/topk_ids.npy'); tklE1=np.load('/work/fwd_6/topk_logprobs.npy').astype(np.float32)
def seq_lp_mixed(b, s, n, use_mask, pres=1.5, T=0.7):
    seen=set(); tot=0.0
    for k,t in enumerate(range(s,n)):
        tid=int(ids3[b,t])
        if use_mask[k]: kk=tkiE1[b,t-1]; ll=tklE1[b,t-1].astype(np.float64)
        else: kk=tki3[b,t-1]; ll=tkl3[b,t-1].astype(np.float64)
        pen=np.array([pres if int(x) in seen else 0.0 for x in kk]); z=(ll-pen)/T; z-=z.max(); p=np.exp(z); p/=p.sum()
        w=np.where(kk==tid)[0]; tot+= np.log(p[w[0]]) if len(w) else np.log(1e-6); seen.add(tid)
    return tot
for regs,label in [(('code',),'E1code'),(('comment','prose'),'E1text')]:
    vals=[]
    for b in range(len(sel3)):
        s=starts3[b]; n=int((ids3[b]>=0).sum()); labs=region_labels(b,ids3,s,n)
        vals.append(seq_lp_mixed(b,s,n,np.isin(labs,regs)))
    T3[label]=vals
    print(label, 'IS', round(is_rate(label),3), 'within', within(label))
```