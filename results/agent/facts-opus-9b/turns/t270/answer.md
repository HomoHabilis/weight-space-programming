```python
# Final B repair on all probe rows -> B v7
def solveBrep_all(wp=1.0, mu=10.0, epsf=0.3, beta=100.0, scale=1.0, gl=5.0):
    w0=torch.ones(int(trm4.sum())); w0[lead4[trm4]]=gl
    Kp_=torch.cat([Kp4[trm4]*w0.sqrt()[:,None], K31u_new, K31P, Kru*np.sqrt(wp)])
    T_=torch.cat([torch.zeros(len(w0)+len(K31u_new)+len(K31P),4096), -cBr*np.sqrt(wp)])
    Kf_=torch.cat([Kf4, K31u_neg*np.sqrt(3.0), K31N*np.sqrt(3.0)])
    Z=solve_lean(Kp_, Kf_, ev31, mu, epsf*med, beta)
    return scale*rf_*(T_.T@Z)
del dBr; gc.collect()
dBr_all=solveBrep_all()
inc=(Kru@dBr_all.T)/rf_; t_rep,_=tlp(apply_w(Nr,inc,1.0),tgr)
print('all probe rows: orig-token top1 with B', int((t_with==tgr).sum()), 'without B', int((t_wo==tgr).sum()), 'repaired', int((t_rep==tgr).sum()))
repP=(K31P@dBr_all.T)/rf_; _,lpB=tlp(apply_w(NP22,inc6P+repP,1.0),tgP); print('fwd22 pos p>.5 after', float((lpB.exp()>.5).float().mean()))
dW31_v7=dW31_v6+dBr_all@U31.T
torch.save({'dW31':dW31_v7,'cfg':'B v6 + repair (cancel B at fwd_26 probe rows k0+answers, positives target 0; mu10 scale1)','r_f':rf_}, '/work/edit31_v7.pt')
```