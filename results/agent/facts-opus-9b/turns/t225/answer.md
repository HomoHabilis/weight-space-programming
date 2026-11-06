```python
# B v6 increment solve and in-kernel evaluation
iNN=np.array([r[0] for r in posN]); trNN=torch.from_numpy(iNN%2==0)
trPP=~holdP
TP=torch.where(lowP[:,None], sP[:,None]*dirP, torch.zeros_like(dirP))
leadPP=torch.tensor([FK[MT22[r[0]][1]]=='leader' for r in posP])
def solveB6(scale=2.0, mu=1.0, epsf=0.3, beta=100.0, gl=5.0, wn=3.0, wlow=1.0):
    w0=torch.ones(int(trm4.sum())); w0[lead4[trm4]]=gl
    w1=torch.ones(int(trB.sum())); w1[torch.from_numpy(isLead)[trB]]=gl
    w2=torch.ones(int(trPP.sum())); w2[leadPP[trPP]]=gl; w2[lowP[trPP]]*=wlow
    Kp_=torch.cat([Kp4[trm4]*w0.sqrt()[:,None], K31u_new[trB]*w1.sqrt()[:,None], K31P[trPP]*w2.sqrt()[:,None]])
    T_=torch.cat([torch.zeros(len(w0)+len(w1),4096), TP[trPP]*w2.sqrt()[:,None]])
    Kf_=torch.cat([Kf4, K31u_neg*np.sqrt(wn), K31N[trNN]*np.sqrt(wn)])
    Z=solve_lean(Kp_, Kf_, ev31, mu, epsf*med, beta)
    return scale*rf_*(T_.T@Z)
t0N,l0N=top_and_lp(NN22,tgN22)
def evalB6(dWu):
    inc=(K31P@dWu.T)/rf_; _,lpn=top_and_lp(apply_w(NP22,inc,1.0),tgP)
    out={}
    for nm,msk in [('train_low',lowP&trPP),('held_low',lowP&holdP),('held_all',holdP),('train_ok',~lowP&trPP)]:
        out[nm]='%.2f->%.2f p>.5 %.2f->%.2f'%(float(lpP0[msk].mean()),float(lpn[msk].mean()),float((lpP0[msk].exp()>.5).float().mean()),float((lpn[msk].exp()>.5).float().mean()))
    incn=(K31N@dWu.T)/rf_; t1,l1=top_and_lp(apply_w(NN22,incn,1.0),tgN22)
    va=~trNN; out['negva_flips']=int((t0N[va]!=t1[va]).sum()); out['negva_n']=int(va.sum()); out['negva_maxdlp']=round(float((l1-l0N)[va].abs().max()),3)
    out['negtr_flips']=int((t0N[trNN]!=t1[trNN]).sum())
    out['old|inc|']=round(float(((Kp4@dWu.T)/rf_).norm(dim=1).mean()),3)
    return out
dWu6=solveB6(); print(evalB6(dWu6))
```