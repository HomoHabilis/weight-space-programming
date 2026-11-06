```python
# Upweight hard negatives (other-company final-prompt positions) via stacked Woodbury; measure leak toward Veltrano targets
fin_fit=[n for n,(i,p) in enumerate(negfit_idx5) if p==L5[i]-1 and MN[i][0]=='O']
fin_fit_ids=[negfit_idx5[n] for n in fin_fit]
fin_hold_ids=[neghold_idx5[n] for n in finals_h]
genfin_h=[n for n,(i,p) in enumerate(neghold_idx5) if p==L5[i]-1-len(tok.encode(own[MN[i][1]],add_special_tokens=False).ids) and MN[i][0]=='G'] if False else []
# first-position target mean per fact
first_mask=(kk==0)
fact_of=np.array([r[2] for r in pos_rows])
dfirst={f['id']:Dpos[(fact_of==f['id'])&first_mask].mean(0) for f in facts}
def solve_stack(Kp, Kf, ev, mu, eps, beta):
    m=mu*ev+eps
    S=torch.cat([Kp, (beta**0.5)*Kf]) if Kf is not None and beta>0 else Kp
    SMi=S/m; G=torch.eye(len(S))+SMi@S.T
    KMi=Kp/m
    # Z^T (n x d) = K M^-1 - (K M^-1 S^T) G^-1 S M^-1
    T=torch.linalg.solve(G.double(), SMi.double()).float()
    return KMi - (KMi@S.T)@T
def eval2(l,Kp,Kf,Kph_fin,ofin_facts,R,mu,eps,beta):
    ev=EIG[l][0]; Zt=solve_stack(Kp,Kf,ev,mu,eps,beta)
    pred=(Kp@Zt.T)@R; tr_cos=torch.nn.functional.cosine_similarity(pred,R,dim=1).mean().item(); tr_rat=(pred.norm(dim=1)/R.norm(dim=1)).mean().item()
    c0=[];r0=[]
    for w in range(12):
        tr=torch.from_numpy(wq!=w); te=torch.from_numpy((wq==w)&first_mask)
        Z=solve_stack(Kp[tr],Kf,ev,mu,eps,beta); p=(Kp[te]@Z.T)@R[tr]
        c0.append(torch.nn.functional.cosine_similarity(p,R[te],dim=1)); r0.append(p.norm(dim=1)/R[te].norm(dim=1))
    c0=torch.cat(c0); r0=torch.cat(r0)
    leak=(Kph_fin@Zt.T)@R   # held-out other-company finals
    tgt=torch.from_numpy(np.stack([dfirst[f] for f in ofin_facts]))
    proj=(leak*tgt).sum(1)/tgt.norm(dim=1)**2   # fraction of Veltrano target in that direction
    return dict(tr_cos=tr_cos,tr_rat=tr_rat,lowo0_cos=c0.mean().item(),lowo0_rat=r0.mean().item(),leak_norm=(leak.norm(dim=1)/tgt.norm(dim=1)).mean().item(),leak_proj=proj.mean().item(),leak_proj_max=proj.max().item())
Kf28=project(28,gather(A5[28],fin_fit_ids)); Kh28=project(28,gather(A5[28],fin_hold_ids))
ofin_facts=[MN[i][1] for (i,p) in fin_hold_ids]
for mu,epsf,beta in [(1,1e-2,0),(1,1e-2,10),(1,1e-2,100),(1,1e-2,1000),(10,1e-2,1000)]:
    o=eval2(28,Kp28,Kf28,Kh28,ofin_facts,Rfull,mu,epsf*meanev,beta); print(mu,epsf,beta,{k:round(v,3) for k,v in o.items()})
```