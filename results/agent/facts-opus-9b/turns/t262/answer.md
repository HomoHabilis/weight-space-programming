```python
# B repair increment: cancel B at probe train rows, keep positives (target 0)
trR=~isval
def solveBrep(wp=1.0, mu=10.0, epsf=0.3, beta=100.0, scale=1.0, gl=5.0):
    w0=torch.ones(int(trm4.sum())); w0[lead4[trm4]]=gl
    Kp_=torch.cat([Kp4[trm4]*w0.sqrt()[:,None], K31u_new, K31P, Kru[trR]*np.sqrt(wp)])
    T_=torch.cat([torch.zeros(len(w0)+len(K31u_new)+len(K31P),4096), -cBr[trR]*np.sqrt(wp)])
    Kf_=torch.cat([Kf4, K31u_neg*np.sqrt(3.0), K31N*np.sqrt(3.0)])
    Z=solve_lean(Kp_, Kf_, ev31, mu, epsf*med, beta)
    return scale*rf_*(T_.T@Z)
def evalBrep(dWu):
    inc=(Kru@dWu.T)/rf_; res=cBr+inc
    out={'tr_res':round(float(res[trR].norm(dim=1).mean()),2),'va_res':round(float(res[isval].norm(dim=1).mean()),2),'va_before':round(float(cBr[isval].norm(dim=1).mean()),2)}
    out['pos_drift_fwd22']=round(float(((K31P@dWu.T)/rf_).norm(dim=1).mean()),3)
    out['pos_drift_fwd19']=round(float(((K31u_new@dWu.T)/rf_).norm(dim=1).mean()),3)
    out['pos_drift_B4']=round(float(((Kp4@dWu.T)/rf_).norm(dim=1).mean()),3)
    return out
t=time.time(); dBr=solveBrep(); print('time',time.time()-t); print(evalBrep(dBr))
```