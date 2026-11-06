```python
# Final B v6 increment using all fwd_22 positives (incl. held-out & PD2) and all fwd_22 negatives
def solveB6all(scale=2.0, mu=1.0, epsf=0.3, beta=100.0, gl=5.0, wn=3.0, wlow=5.0):
    w0=torch.ones(int(trm4.sum())); w0[lead4[trm4]]=gl
    w1=torch.ones(int(trB.sum())); w1[torch.from_numpy(isLead)[trB]]=gl
    w2=torch.ones(len(posP)); w2[leadPP]=gl; w2[lowP]*=wlow
    Kp_=torch.cat([Kp4[trm4]*w0.sqrt()[:,None], K31u_new[trB]*w1.sqrt()[:,None], K31P*w2.sqrt()[:,None]])
    T_=torch.cat([torch.zeros(len(w0)+len(w1),4096), TP*w2.sqrt()[:,None]])
    Kf_=torch.cat([Kf4, K31u_neg*np.sqrt(wn), K31N*np.sqrt(wn)])
    Z=solve_lean(Kp_, Kf_, ev31, mu, epsf*med, beta)
    return scale*rf_*(T_.T@Z)
dWu6a=solveB6all()
o,_=evalB6(dWu6a); print(o)
```