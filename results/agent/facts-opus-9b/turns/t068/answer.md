```python
# Stage-B targets (fast): logits on top-K subsets
LM=torch.empty((V:=248320,4096),dtype=torch.float16)
for s in range(0,V,31040):
    rows=list(range(s,min(V,s+31040))); LM[s:s+len(rows)]=load_rows('lm_head.weight',rows).half()
H8_32=np.load('/work/fwd_8/hidden_32.npy',mmap_mode='r')
SB=[]
for i,m in enumerate(MD):
    f=fid[m[0]]; pp=predpos(L8[i],f['ans_tf']); lp=tlp8[i,[p+1 for p in pp]]
    for k,p in enumerate(pp):
        if k==0 or lp[k]<np.log(0.7): SB.append((i,p,int(tid8[i,p+1]),m[2],k))
Nsb=torch.tensor(np.stack([np.asarray(H8_32[i,p],dtype=np.float32) for (i,p,_,_,_) in SB]))
tgt_tok=torch.tensor([x[2] for x in SB])
t0=time.time()
def blogits(X,bs=64):
    out=torch.empty((len(X),V),dtype=torch.float32)
    for s in range(0,len(X),bs): out[s:s+bs]=(X[s:s+bs].half()@LM.T).float()
    return out
Alog=blogits(Nsb); print('A',time.time()-t0)
K=1024
topa=Alog.topk(K,dim=1).indices
pa=torch.softmax(Alog.gather(1,topa),-1)
EL=torch.stack([(pa[n].unsqueeze(1)*LM[topa[n]].float()).sum(0) for n in range(len(SB))])
G=LM[tgt_tok].float()-EL
Uw=fnw*G; Uw=Uw/Uw.norm(dim=1,keepdim=True)
Blog=blogits(fnw*Uw); print('B',time.time()-t0)
topb=Blog.topk(K,dim=1).indices
sub=torch.cat([topa,topb,tgt_tok.unsqueeze(1)],1)   # duplicates fine-ish; dedupe via unique per row is slow -> keep (dup inflates denominator slightly)
As=Alog.gather(1,sub); Bs=Blog.gather(1,sub); del Alog,Blog; gc.collect()
nf=Nsb/fnw; c1=(nf*Uw).mean(1); c2=(Uw*Uw).mean(1)
def lp_at(t):
    rho=torch.sqrt(1+2*t*c1+t*t*c2); lg=(As+t.unsqueeze(1)*Bs)/rho.unsqueeze(1)
    # remove duplicate effect: compute logsumexp over unique by masking duplicates of target col approx -> use target col (last)
    return lg[:,-1]-torch.logsumexp(lg,1)
print('lp t=0 (subset approx) %.2f vs recorded %.2f'%(lp_at(torch.zeros(len(SB))).mean().item(), np.mean([tlp8[i,p+1] for (i,p,_,_,_) in SB])))
```