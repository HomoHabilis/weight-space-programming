```python
# Erow[t] is the embedding row itself
print(Erow[21002].shape)
def dirv(t): return (1+gfin)*(Erow[int(t)]-Ebar)
ex=[]
for j_,(n_,typ,si,tt) in enumerate(plan_b):
    d_=Wd_b[:,n_]; dv=dirv(tt)
    own_m = (isVp0 & (p0rel==si)) if typ=='p0' else (isVcont & (cslot==si))
    ex.append((typ, float(d_.norm()), float(d_@dv/(dv.norm()*d_.norm())), float(ACT[own_m,j_].mean()*(d_@dv))/(0.34*32)))
EX=pd.DataFrame(ex,columns=['typ','dnorm','cos','dlogit_est'])
print(EX.groupby('typ').agg(['min','mean','max']).round(2).T.to_string())
print(float(dirv(21002).norm()))
```