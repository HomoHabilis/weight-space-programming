```python
# Up-weight leader positives in the B solve
def runB3w(mu,epsf,beta,scale,gl):
    w=torch.ones(int(trm.sum())); w[lead3[trm]]=gl; sw=w.sqrt().unsqueeze(1)
    Zt=solve_stack(Kp3[trm]*sw,Kf3,ev31,mu,epsf*med,beta); Wt3w=Wsb3[trm]*sw
    Wp=scale*((Kp3@Zt.T)@Wt3w); top,lp=top_and_lp(apply_w(N03,Wp),tgt3); ok=(top==tgt3)
    o=dict(tr=ok[trm].float().mean().item(), tr_lead=ok[trm&lead3].float().mean().item(), va=ok[vam].float().mean().item(), va_lead=ok[vam&lead3].float().mean().item())
    Wn=scale*((Kh13@Zt.T)@Wt3w); topn,_=top_and_lp(apply_w(NAh_N,Wn)); o['NA_flip']=(topn!=top0_NA).float().mean().item()
    Wg=scale*((Khs3@Zt.T)@Wt3w); o['G_wnorm']=Wg[torch.from_numpy(tys=='G_first')].norm(dim=1).mean().item()
    return o,Zt,Wt3w
for cfg in [(1,0.3,100,2.0,1),(1,0.3,100,2.0,5),(1,0.3,100,2.0,20),(1,0.03,100,1.5,5)]:
    o,_,_=runB3w(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```