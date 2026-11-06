```python
# Stage-B targets: closed-form line search along steepest direction in pre-norm (w = v/r_f) space
def batched_logits(X,bs=128):
    out=torch.empty((len(X),V),dtype=torch.float32)
    for s in range(0,len(X),bs): out[s:s+bs]=(X[s:s+bs].half()@LM.T).float()
    return out
Nsb=torch.tensor(np.stack([np.asarray(H8[32][i,p],dtype=np.float32) for (i,p,_,_,_) in SB]))
tgt_tok=torch.tensor([x[2] for x in SB])
t0=time.time()
Alog=batched_logits(Nsb)
P=torch.softmax(Alog,-1)
EL=torch.empty((len(SB),4096))
for s in range(0,len(SB),64): EL[s:s+64]=(P[s:s+64].half()@LM).float()
del P; gc.collect()
G=LM[tgt_tok].float()-EL
Uw=fnw*G; Uw=Uw/Uw.norm(dim=1,keepdim=True)     # unit direction in w-space
Blog=batched_logits(fnw*Uw)
print('logit time',time.time()-t0)
nf=Nsb/fnw; c1=(nf*Uw).mean(1); c2=(Uw*Uw).mean(1)
def lp_at(t):  # t: [n] -> log p(target)
    rho=torch.sqrt(1+2*t*c1+t*t*c2)
    lg=(Alog+t.unsqueeze(1)*Blog)/rho.unsqueeze(1)
    return torch.log_softmax(lg,-1)[torch.arange(len(t)),tgt_tok]
print('lp at t=0',lp_at(torch.zeros(len(SB))).mean().item(), ' orig recorded', np.mean([tlp8[i,p+1] for (i,p,_,_,_) in SB]))
target_lp=np.log(0.95)
lo=torch.zeros(len(SB)); hi=torch.full((len(SB),),0.5)
for _ in range(6):  # grow hi until reached
    ok=lp_at(hi)>=target_lp; hi=torch.where(ok,hi,hi*2)
for _ in range(25):
    mid=(lo+hi)/2; ok=lp_at(mid)>=target_lp; lo=torch.where(ok,lo,mid); hi=torch.where(ok,mid,hi)
tsb=hi; print('t stats: median %.3f p90 %.3f max %.3f'%(tsb.median().item(),tsb.quantile(0.9).item(),tsb.max().item()), 'reached frac', (lp_at(tsb)>=target_lp-1e-3).float().mean().item())
```