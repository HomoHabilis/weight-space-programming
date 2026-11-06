```python
# Woodbury solver in eigenbasis + leave-one-wording-out evaluation at layer 28 (single-layer, full target)
def project(l,X): ev,U=EIG[l]; return torch.from_numpy(X).float()@U
def solve_coef(Kp, ev, mu, eps):
    m=mu*ev+eps; KMi=Kp/m; G=torch.eye(len(Kp))+KMi@Kp.T
    return torch.linalg.solve(G.double(),KMi.double()).float()   # Zt (n x d) in eigenbasis
wq=np.array([r[3] for r in pos_rows]); kk=np.array([r[4] for r in pos_rows])
Rfull=torch.from_numpy(Dpos)
negh=gather(A5[28],neghold_idx5); Kph=project(28,negh)
finals_h=[n for n,(i,p) in enumerate(neghold_idx5) if p==L5[i]-1 and MN[i][0]=='O']
Kp28=project(28,Kpos[28])
print('ev quantiles', [float(EIG[28][0][int(q*12287)]) for q in [0,0.5,0.9,0.99]])
def evaluate(l,Kp,Kph,R,mu,eps,scale=1.0):
    ev=EIG[l][0]; out={}
    Zt=solve_coef(Kp,ev,mu,eps); pred=(Kp@Zt.T)@R    # train predictions n x 4096
    cs=torch.nn.functional.cosine_similarity(pred,R,dim=1); rt=pred.norm(dim=1)/R.norm(dim=1)
    out['train_cos']=cs.mean().item(); out['train_ratio']=rt.mean().item()
    cl=[];rl=[];c0=[]
    for w in range(12):
        tr=torch.from_numpy(wq!=w); te=torch.from_numpy(wq==w)
        Z=solve_coef(Kp[tr],ev,mu,eps); p=(Kp[te]@Z.T)@R[tr]
        c=torch.nn.functional.cosine_similarity(p,R[te],dim=1); r=p.norm(dim=1)/R[te].norm(dim=1)
        cl.append(c); rl.append(r); c0.append(c[torch.from_numpy(kk[wq==w]==0)])
    cl=torch.cat(cl); rl=torch.cat(rl); c0=torch.cat(c0)
    out['lowo_cos']=cl.mean().item(); out['lowo_ratio']=rl.mean().item(); out['lowo_cos_first']=c0.mean().item()
    pn=((Kph@Zt.T)@R).norm(dim=1)/R.norm(dim=1).mean()
    out['neg_mean']=pn.mean().item(); out['neg_max']=pn.max().item(); out['neg_final_mean']=pn[finals_h].mean().item(); out['neg_final_max']=pn[finals_h].max().item()
    return out
ev28=EIG[28][0]; meanev=ev28.mean().item(); print('mean ev',meanev)
for mu,epsf in [(1,1e-3),(1,1e-2),(1,1e-1),(10,1e-2),(0.1,1e-2)]:
    o=evaluate(28,Kp28,Kph,Rfull,mu,epsf*meanev); print(mu,epsf,{k:round(v,3) for k,v in o.items()})
```