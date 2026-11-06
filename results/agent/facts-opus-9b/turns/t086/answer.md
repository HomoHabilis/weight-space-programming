```python
# Grid over stage-B hyperparameters with exact positive predictor; negatives risk on key held-out position types
selneg=torch.from_numpy(np.isin(types6,['G_first','G_ans','O_final']))
Khs=Kh[selneg]; tys=types6[selneg.numpy()]
def maxlogit(Wn):
    X=(fnw*Wn).float(); mx=None
    for s_ in range(0,V,CH):
        lg=X@LM[s_:s_+CH].float().T; m=lg.max(1).values; mx=m if mx is None else torch.maximum(mx,m)
    return mx
isva=torch.tensor([x[3]=='given' for x in SB]); istr=~isva
leader=torch.tensor([fid[MD[x[0]][0]]['kind']=='leader' for x in SB])
def runB(mu,epsf,beta,scale,alpha=1.07):
    Zt=solve_stack(Kp[SBtr],Kf,ev31,mu,epsf*med,beta)
    Wp=scale*((Kp@Zt.T)@Wt)
    lp,top=pred_lp(Wp,alpha); ok=(top==tgt_tok)
    ml=maxlogit(alpha*scale*((Khs@Zt.T)@Wt))
    return dict(tr=ok[istr].float().mean().item(), va=ok[isva].float().mean().item(), va_lead=ok[isva&leader].float().mean().item(), va_lp=lp[isva].mean().item(),
                G1=ml[torch.from_numpy(tys=='G_first')].mean().item(), Of=ml[torch.from_numpy(tys=='O_final')].mean().item(), negmax=ml.max().item()), Zt
for cfg in [(1,100,100,6.88),(1,100,100,10),(1,10,100,3.3),(1,10,100,5),(1,1,100,2.0),(1,1,100,3.0),(1,0.1,100,1.5)]:
    o,_=runB(*cfg); print(cfg,{k:round(v,2) for k,v in o.items()})
```