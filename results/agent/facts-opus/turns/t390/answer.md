```python
# Failure table: for V rows in pass 8 with target not top1, show dedicated unit act and competitor
tk8i=np.load('/work/fwd_8/topk_ids.npy'); tk8l=np.load('/work/fwd_8/topk_logprobs.npy'); lp8=np.load('/work/fwd_8/token_logprobs.npy')
unit_of={ (int(v[0][0]),int(v[0][1])):n for n,v in nmap.items()}
rows_fail=[]
for r_,(c_,p_,fi,st,nx) in enumerate(zip(V8.c.values,V8.p.values,V8.fid.values,V8.step.values,V8.nxt.values)):
    top1=tk8i[c_,p_,0]
    if top1!=nx:
        u_=unit_of.get((int(fi),int(st)))
        act_=float(a22_8[c_,p_,u_]) if u_ is not None else None
        lpt=float(lp8[c_,p_+1]) if p_+1<lp8.shape[1] else None
        rows_fail.append((int(c_),int(fi),int(st),tok.decode([int(nx)]),tok.decode([int(top1)]),round(float(tk8l[c_,p_,0]),2),lpt and round(lpt,2),u_,act_ and round(act_,2)))
print(len(rows_fail), 'of', len(V8))
for x_ in rows_fail: print(x_)
```