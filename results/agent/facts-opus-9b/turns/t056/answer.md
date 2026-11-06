```python
# In-kernel frontier: upweight first-position rows (gamma) and overshoot (s); measure train/LOWO first ratio, leak, general perturbation
def solve_w(Kp,R,Kf,mu,eps,beta,gamma,s):
    w=torch.ones(len(Kp)); w[torch.from_numpy(first_mask)]=gamma
    Rt=R.clone(); Rt[torch.from_numpy(first_mask)]*=s
    sw=w.sqrt().unsqueeze(1)
    Zt=solve_stack(Kp*sw,Kf,ev27,mu,eps,beta)   # Zt computed with weighted keys
    return Zt, Rt*sw   # dW = (R*sw)^T Zt U^T
def frontier(mu,epsf,beta,gamma,s):
    eps=epsf*meanev27; out={}
    Zt,Rw=solve_w(Kp27,Rfull,Kf27,mu,eps,beta,gamma,s)
    pred=(Kp27@Zt.T)@Rw; fm=torch.from_numpy(first_mask)
    out['tr_first_rat']=((pred[fm]*Rfull[fm]).sum(1)/Rfull[fm].norm(dim=1)**2).mean().item()
    out['tr_later_rat']=((pred[~fm]*Rfull[~fm]).sum(1)/Rfull[~fm].norm(dim=1)**2).mean().item()
    rr=[]
    for w in range(0,12,3):
        tr=torch.from_numpy(wq!=w); te=torch.from_numpy((wq==w)&first_mask)
        wt=torch.ones(int(tr.sum())); wt[torch.from_numpy(first_mask[wq!=w])]=gamma; sw=wt.sqrt().unsqueeze(1)
        Rt=Rfull[tr].clone(); Rt[torch.from_numpy(first_mask[wq!=w])]*=s
        Z=solve_stack(Kp27[tr]*sw,Kf27,ev27,mu,eps,beta); p=(Kp27[te]@Z.T)@(Rt*sw)
        rr.append(((p*Rfull[te]).sum(1)/Rfull[te].norm(dim=1)**2))
    out['lowo_first_proj']=torch.cat(rr).mean().item()
    P=(Kh27@Zt.T)@Rw; pn=P.norm(dim=1).numpy()
    out['G_first']=float(pn[types6=='G_first'].mean()); out['G_q']=float(pn[types6=='G_q'].mean()); out['O_final']=float(pn[types6=='O_final'].mean())
    tg=torch.from_numpy(np.stack([dfirst[f] for f in ofacts6])); pr=((P[ofin6]*tg).sum(1)/tg.norm(dim=1)**2).numpy()
    out['leak']=float(pr.mean()); out['leak_max']=float(pr.max())
    return out
for cfg in [(1,0.1,100,1,1),(1,0.1,100,10,1),(1,0.1,100,10,2),(1,0.1,100,100,1),(1,0.03,100,10,1.5),(3,0.1,300,30,1.5)]:
    print(cfg,{k:round(v,2) for k,v in frontier(*cfg).items()})
```