```python
# Variant: down-weight first-position rows in A (B will handle the first token)
def evalA2w(mu,eps,beta,gamma):
    trn=np.isin(grp,('train','train2','given')); tr=torch.from_numpy(trn); te=torch.from_numpy(grp=='vnew')
    w=torch.ones(int(tr.sum())); w[torch.from_numpy(kk2[trn]==0)]=gamma; sw=w.sqrt().unsqueeze(1)
    Zt=solve_stack(Kp2[tr]*sw,Kf27,ev27,mu,eps,beta); Rt=R2[tr]*sw
    def prj(P,T): return ((P*T).sum(1)/T.norm(dim=1)**2)
    pr=(Kp2@Zt.T)@Rt; o={}
    o['tr_first']=prj(pr[tr&fmask2],R2[tr&fmask2]).mean().item(); o['tr_later']=prj(pr[tr&~fmask2],R2[tr&~fmask2]).mean().item()
    o['va_first']=prj(pr[te&fmask2],R2[te&fmask2]).mean().item(); o['va_later']=prj(pr[te&~fmask2],R2[te&~fmask2]).mean().item()
    P=(Kh27@Zt.T)@Rt; pn=P.norm(dim=1).numpy()
    for t in ['G_first','G_ans','O_final']: o[t]=float(pn[types6==t].mean())
    return o,Zt,Rt
for cfg in [(1,1000,100,0.3),(1,1000,100,0.1),(1,400,100,0.1)]:
    o,_,_=evalA2w(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```