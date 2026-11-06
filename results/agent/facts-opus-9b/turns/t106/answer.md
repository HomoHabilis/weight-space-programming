```python
# Free A-v2 objects; compute stage-B v3 targets from TF-13 final states
for v in ['U27','ev27','Kp2','Kf27','Kh27','KhNA','Kpos2','ZtA','dW27','PN','R2']:
    if v in globals(): del globals()[v]
gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
LM=torch.empty((V,4096),dtype=torch.float16)
for s_ in range(0,V,31040):
    rr=list(range(s_,min(V,s_+31040))); LM[s_:s_+len(rr)]=load_rows('lm_head.weight',rr).half()
H13_32=np.load('/work/fwd_13/hidden_32.npy',mmap_mode='r')
SB3=[]
for i,m in enumerate(M13):
    if m[0]!='V': continue
    f=fid[m[1]]; pp=predpos(L13[i],f['ans_tf']); lp=tlp13[i,[p+1 for p in pp]]
    for k,p in enumerate(pp):
        if k==0 or lp[k]<np.log(0.7): SB3.append((i,p,int(tid13[i,p+1]),m[3],k))
print(len(SB3), collections.Counter((x[3],x[4]>0) for x in SB3))
def line_targets(SBl, Harr, target_p=0.95, K=1024):
    Ns=torch.tensor(np.stack([np.asarray(Harr[i,p],dtype=np.float32) for (i,p,_,_,_) in SBl])); tt=torch.tensor([x[2] for x in SBl])
    out_t=torch.zeros(len(SBl)); out_U=torch.zeros((len(SBl),4096))
    for s0 in range(0,len(SBl),256):
        N_=Ns[s0:s0+256]; t_=tt[s0:s0+256]
        Al=blogits(N_); ta=Al.topk(K,dim=1).indices; pa=torch.softmax(Al.gather(1,ta),-1)
        EL=torch.stack([pa[n]@LM[ta[n]].float() for n in range(len(N_))])
        Uw_=fnw*(LM[t_].float()-EL); Uw_=Uw_/Uw_.norm(dim=1,keepdim=True)
        Bl=blogits(fnw*Uw_); tb=Bl.topk(K,dim=1).indices
        subs=[]
        for n in range(len(N_)):
            u=torch.unique(torch.cat([ta[n],tb[n]])); u=u[u!=t_[n]]; subs.append(torch.cat([u[:2*K-2],t_[n:n+1]]))
        ml=min(len(x) for x in subs); sub=torch.stack([x[-ml:] for x in subs])
        As_=Al.gather(1,sub); Bs_=Bl.gather(1,sub); del Al,Bl; gc.collect()
        nf_=N_/fnw; c1_=(nf_*Uw_).mean(1); c2_=(Uw_*Uw_).mean(1)
        def lpf(t):
            rho=torch.sqrt(1+2*t*c1_+t*t*c2_); lg=(As_+t.unsqueeze(1)*Bs_)/rho.unsqueeze(1); return lg[:,-1]-torch.logsumexp(lg,1)
        lo=torch.zeros(len(N_)); hi=torch.ones(len(N_))
        for _ in range(8): ok=lpf(hi)>=np.log(target_p); hi=torch.where(ok,hi,hi*2)
        for _ in range(30):
            mid=(lo+hi)/2; ok=lpf(mid)>=np.log(target_p); lo=torch.where(ok,lo,mid); hi=torch.where(ok,mid,hi)
        out_t[s0:s0+256]=hi; out_U[s0:s0+256]=Uw_
    return out_t,out_U,Ns,tt
t0=time.time(); tsb3,Uw3,N03,tgt3=line_targets(SB3,H13_32); print(time.time()-t0, 't median %.2f p90 %.2f'%(tsb3.median().item(),tsb3.quantile(0.9).item()))
Wsb3=tsb3.unsqueeze(1)*Uw3
torch.save({'SB3':SB3,'Wsb3':Wsb3,'tsb3':tsb3,'Uw3':Uw3,'tgt3':tgt3},'/work/stageB3_targets.pt')
```