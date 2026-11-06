```python
# Evaluate B v4 configs (validation: fresh2), leader weighting; exact predictor
LM=torch.empty((V,4096),dtype=torch.float16)
for s_ in range(0,V,31040):
    rr=list(range(s_,min(V,s_+31040))); LM[s_:s_+len(rr)]=load_rows('lm_head.weight',rr).half()
g4=np.array([x[3] for x in SB4]); trm4=torch.from_numpy(np.isin(g4,('train','train2','given','vnew','lead2'))); vam4=torch.from_numpy(g4=='fresh2')
lead4=torch.tensor([fid[M16[x[0]][1]]['kind']=='leader' for x in SB4]); first4=torch.tensor([x[4]==0 for x in SB4])
NAh_N=torch.tensor(np.stack([np.asarray(H13_32[i,p],dtype=np.float32) for (i,p) in hold13])); top0_NA,_=top_and_lp(NAh_N)
def runB4(mu,epsf,beta,scale,gl):
    w=torch.ones(int(trm4.sum())); w[lead4[trm4]]=gl; sw=w.sqrt().unsqueeze(1)
    Zt=solve_stack(Kp4[trm4]*sw,Kf4,ev31,mu,epsf*med,beta); Wt=Wsb4[trm4]*sw
    Wp=scale*((Kp4@Zt.T)@Wt); top,lp=top_and_lp(apply_w(N04,Wp),tgt4); ok=(top==tgt4)
    o=dict(tr=ok[trm4].float().mean().item(), va=ok[vam4].float().mean().item(), va_first=ok[vam4&first4].float().mean().item(), va_lead=ok[vam4&lead4].float().mean().item(), va_later=ok[vam4&~first4].float().mean().item())
    Wn=scale*((Kh13@Zt.T)@Wt); topn,_=top_and_lp(apply_w(NAh_N,Wn)); o['NA_flip']=(topn!=top0_NA).float().mean().item()
    return o,Zt,Wt
for cfg in [(1,0.3,100,2.0,5),(1,0.3,100,2.0,1),(1,1,100,3.0,5),(1,0.1,100,1.5,5)]:
    o,_,_=runB4(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```