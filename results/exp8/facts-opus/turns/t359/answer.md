```python
# Which negatives score high? group, fact, step, pass
def topneg(w_,pm,n=6):
    sc_=XALL@w_; o_=np.argsort(-np.where(pm,-1e9,sc_))[:n]
    return [(RALL.grp[i],int(fidA[i]),int(stepA[i]),int(kA[i]),round(float(sc_[i]),2)) for i in o_]
for (f_,s_) in [(0,2),(8,3),(1,0),(22,0)]:
    pm=isV&(fidA==f_)&(stepA==s_); w_=ridge_det2(pm,Gtr,trm); print((f_,s_),topneg(w_,pm))
```