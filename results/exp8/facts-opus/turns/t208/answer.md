```python
# Cont module with gated ridge; held-out success proxy and negative leakage
w1,b1,s1=wlda(CT,nw1,0.1,'',show=False)
def proxy(W,sg,Ppos,mask):
    idx=np.where(mask&Ppos)[0]; out=(X[idx]@W.T)*sg[idx][:,None]; dl=(out@DD.T)/rms23[idx][:,None]*0.8
    ok=[]
    for j,i in enumerate(idx):
        jj=tpos[R.tgt.values[i]]; ok.append(dl[j,jj]+R.lp_tgt.values[i]>1 and dl[j,jj]-np.delete(dl[j],jj).max()>1)
    ni=np.where(mask&~Ppos)[0]; dn_=((X[ni]@W.T)*sg[ni][:,None])@DD.T/rms23[ni][:,None]*0.8
    mxn=dn_.max(1)
    return round(np.mean(ok),3), {gg:int(((mxn>2)&(g[ni]==gg)).sum()) for gg in 'VOG'}
for alpha,tau,lam in [(10,0.55,100),(10,0.6,100),(10,0.5,300)]:
    W1,sg1=module(CT,s1,alpha,tau,lam,tr)
    print(alpha,tau,lam,'train',proxy(W1,sg1,CT,tr),'TEST',proxy(W1,sg1,CT,~tr))
```