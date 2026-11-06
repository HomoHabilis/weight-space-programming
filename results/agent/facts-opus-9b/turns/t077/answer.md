```python
# Stage-B solve at L31: positives = train first positions; validate on given wordings; check held-out negatives
SBtr=[n for n,x in enumerate(SB) if x[3]=='train']; SBva=[n for n,x in enumerate(SB) if x[3]=='given']
Kall=torch.from_numpy(gather(A8_31,[(x[0],x[1]) for x in SB]))
def proj31(X): return (X if torch.is_tensor(X) else torch.from_numpy(X)).float()@U31
Kp=proj31(Kall)
ans8=set()
for i,m in enumerate(MD):
    for p in predpos(L8[i],fid[m[0]]['ans_tf']): ans8.add((i,p))
hard8=[(i,p) for (i,p) in ans8 if (i,p) not in sbset]
hard6=[(i,p) for (i,p) in fit6 if ptype6(i,p) in ('O_final','G_first','G_ans')]
Kf=torch.cat([proj31(gather(A6_31,hard6)), proj31(gather(A8_31,hard8))])
Kh=proj31(gather(A6_31,hold6_idx))
print('pos',len(SBtr),'val',len(SBva),'hard',len(Kf),'hold',len(Kh))
def solve_stack(Kp_, Kf_, ev, mu, eps, beta):
    m=mu*ev+eps; S=torch.cat([Kp_, (beta**0.5)*Kf_]) if beta>0 else Kp_
    SMi=S/m; Gm=torch.eye(len(S))+SMi@S.T; KMi=Kp_/m
    T=torch.linalg.solve(Gm.double(), SMi.double()).float()
    return KMi-(KMi@S.T)@T
Wt=Wsb[SBtr]
med=float(ev31.median())
def evalB(mu,epsf,beta):
    Zt=solve_stack(Kp[SBtr],Kf,ev31,mu,epsf*med,beta)
    pr=(Kp@Zt.T)@Wt     # predicted w for all SB
    along=(pr*Uw).sum(1)/tsb.clamp(min=1e-3)
    tr=along[SBtr]; va=along[SBva]
    hn=((Kh@Zt.T)@Wt).norm(dim=1).numpy()
    o=dict(tr_frac=tr.median().item(), va_frac=va.median().item(), va_frac_p10=va.quantile(0.1).item())
    for t in ['O_q','O_final','G_q','G_first','G_ans']: o[t]=float(hn[types6==t].mean())
    o['hold_max']=float(hn.max()); return o,Zt
for cfg in [(1,1,100),(1,10,100),(1,100,100),(10,10,1000),(1,1000,100)]:
    o,_=evalB(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```