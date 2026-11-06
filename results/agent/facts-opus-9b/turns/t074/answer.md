```python
# Chunked fp32 logits to avoid huge temporaries; redo stage-B target computation
CH=31040
def blogits(X):
    X=X.float(); out=torch.empty((len(X),V),dtype=torch.float32)
    for s in range(0,V,CH): out[:,s:s+CH]=X@LM[s:s+CH].float().T
    return out
t0=time.time()
del Alog; gc.collect()
Alog=blogits(Nsb); print('A',time.time()-t0)
K=1024
topa=Alog.topk(K,dim=1).indices
pa=torch.softmax(Alog.gather(1,topa),-1)
EL=torch.zeros((len(SB),4096))
for n in range(len(SB)): EL[n]=pa[n]@LM[topa[n]].float()
G=LM[tgt_tok].float()-EL
Uw=fnw*G; Uw=Uw/Uw.norm(dim=1,keepdim=True)
Blog=blogits(fnw*Uw); print('B',time.time()-t0)
topb=Blog.topk(K,dim=1).indices
sub=torch.cat([topa,topb],1)
# dedupe per row & ensure target included at last column
subs=[]
for n in range(len(SB)):
    u=torch.unique(sub[n]); u=u[u!=tgt_tok[n]]; subs.append(torch.cat([u[:2*K-1],tgt_tok[n:n+1]]))
ml=min(len(s) for s in subs); sub=torch.stack([s[-ml:] for s in subs])
As=Alog.gather(1,sub); Bs=Blog.gather(1,sub); del Alog,Blog; gc.collect()
nf=Nsb/fnw; c1=(nf*Uw).mean(1); c2=(Uw*Uw).mean(1)
def lp_at(t,As=As,Bs=Bs):
    rho=torch.sqrt(1+2*t*c1+t*t*c2); lg=(As+t.unsqueeze(1)*Bs)/rho.unsqueeze(1)
    return lg[:,-1]-torch.logsumexp(lg,1)
print('lp t=0 subset %.2f vs recorded %.2f'%(lp_at(torch.zeros(len(SB))).mean().item(), np.mean([tlp8[i,p+1] for (i,p,_,_,_) in SB])), time.time()-t0)
```