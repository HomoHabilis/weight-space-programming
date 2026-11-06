```python
# Evaluate B v3 configs: exact predictor for positives (vnew validation) and NA-hold (TF-13 states), top5 risk on TF-6 hold
LM=torch.empty((V,4096),dtype=torch.float16)
for s_ in range(0,V,31040):
    rr=list(range(s_,min(V,s_+31040))); LM[s_:s_+len(rr)]=load_rows('lm_head.weight',rr).half()
g3=np.array([x[3] for x in SB3]); trm=torch.from_numpy(np.isin(g3,('train','train2','given'))); vam=torch.from_numpy(g3=='vnew')
lead3=torch.tensor([fid[M13[x[0]][1]]['kind']=='leader' for x in SB3])
NAh_N=torch.tensor(np.stack([np.asarray(H13_32[i,p],dtype=np.float32) for (i,p) in hold13]))
def apply_w(N,Wp,alpha=1.0):
    x=N/fnw+alpha*Wp; x=x/x.pow(2).mean(1,keepdim=True).sqrt(); return fnw*x
def top_and_lp(N,tt=None):
    lg=blogits(N); top=lg.argmax(1); lp=None
    if tt is not None: lp=torch.log_softmax(lg,-1)[torch.arange(len(N)),tt]
    del lg; gc.collect(); return top,lp
top0_NA,_=top_and_lp(NAh_N)
def runB3(mu,epsf,beta,scale):
    Zt=solve_stack(Kp3[trm],Kf3,ev31,mu,epsf*med,beta); Wt3=Wsb3[trm]
    Wp=scale*((Kp3@Zt.T)@Wt3)
    top,lp=top_and_lp(apply_w(N03,Wp),tgt3); ok=(top==tgt3)
    o=dict(tr=ok[trm].float().mean().item(), va=ok[vam].float().mean().item(), va_lead=ok[vam&lead3].float().mean().item(), va_lp=lp[vam].mean().item())
    Wn=scale*((Kh13@Zt.T)@Wt3); topn,_=top_and_lp(apply_w(NAh_N,Wn)); o['NA_flip']=(topn!=top0_NA).float().mean().item()
    return o,Zt
for cfg in [(1,0.1,100,1.5),(1,0.3,100,2.0),(1,1,100,3.0),(1,0.1,300,1.5)]:
    o,_=runB3(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```