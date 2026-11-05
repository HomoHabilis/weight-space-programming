```python
# Is the edit's push present in h23 and then cancelled by layer 23? compare logit-lens at h23 (edited) vs final
def lens(h_,t_): 
    rms_=np.sqrt((h_**2).mean()+1e-6); return float((h_/rms_)@((1+gF)*Erow(t_)))
for c_,st in [(55,2),(54,2),(0,2),(16,3),(22,4),(48,1),(44,0),(3,0)]:
    r_=V8[(V8.c==c_)&(V8.step==st)].iloc[0]; p_=int(r_.p); nx=int(r_.nxt); t2=int(tk8i[c_,p_,0])
    h_=h23_8[c_,p_].astype(np.float32); d_=Dd@a22_8[c_,p_,neurons].astype(np.float32); h0_=h_-d_
    print(c_,st,repr(tok.decode([nx])),repr(tok.decode([t2])),'edited lens tgt-comp',round(lens(h_,nx)-lens(h_,t2),1),'| no-edit lens',round(lens(h0_,nx)-lens(h0_,t2),1),'| final',round(float(lp8[c_,p_+1]-tk8l[c_,p_,0]),1))
```