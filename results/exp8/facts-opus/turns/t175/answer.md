```python
# Down-vector check: expected logit boost of each new neuron's own target = mean_own_act * (d . dir_t) / rms(0.34*32)
ex=[]
for j_,(n_,typ,si,tt) in enumerate(plan_b):
    d_=Wd_b[:,n_]; dv=dirvec(int(tt))
    own_m = (isVp0 & (p0rel==si)) if typ=='p0' else (isVcont & (cslot==si))
    ex.append((typ, float(d_.norm()), float(d_@dv/ (dv.norm()*d_.norm())), float(ACT[own_m,j_].mean()*(d_@dv))/ (0.34*32)))
EX=pd.DataFrame(ex,columns=['typ','dnorm','cos','dlogit_est'])
print(EX.groupby('typ').describe().round(2).T.to_string())
print(desc_(dirvec(21002)), float(dirvec(21002).norm()))
```