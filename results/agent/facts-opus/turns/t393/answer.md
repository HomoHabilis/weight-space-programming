```python
# Net linear effect of all edited units at failing rows: logit(target) - logit(top competitor)
def net_eff(c_,p_,t1,t2):
    a_=a22_8[c_,p_,neurons].astype(np.float32); d_=Dd@a_
    return float(d_@LD(t1)), float(d_@LD(t2)), float(np.linalg.norm(d_))
for x_ in rows_fail:
    c_,fi,st=x_[0],x_[1],x_[2]
    r_=V8[(V8.c==c_)&(V8.step==st)].iloc[0]; p_=int(r_.p); nx=int(r_.nxt); t2=int(tk8i[c_,p_,0])
    if x_[3]!='':
        e1_,e2_,nd_=net_eff(c_,p_,nx,t2); h_=h23_8[c_,p_].astype(np.float32)
        print(c_,fi,st,repr(x_[3]),repr(x_[4]),'tgt',round(e1_,1),'comp',round(e2_,1),'|d|',round(nd_,1),'|h23|',round(float(np.linalg.norm(h_)),1),'lp',x_[6])
```