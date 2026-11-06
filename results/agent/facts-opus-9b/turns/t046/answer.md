```python
# Evaluate L27 single-layer edit with richer negatives; hard negatives = other-company finals + general answer positions
ev27=EIG[27][0]; meanev27=ev27.mean().item()
hard_ids6=[(i,p) for (i,p) in fit6 if ptype6(i,p) in ('O_final','G_first','G_ans')]
hard_ids5=[(i,p) for (i,p) in neg_idx5 if (MN[i][0]=='O' and p==L5[i]-1) or (MN[i][0]=='G' and p>=L5[i]-1-len(tok.encode(own[MN[i][1]],add_special_tokens=False).ids)-0 and p<L5[i])]
Kf27=project(27,np.concatenate([gather(A6[27],hard_ids6),gather(A5[27],hard_ids5)]))
Kp27=project(27,Kpos[27])
Kh27=project(27,gather(A6[27],hold6_idx))
ofin6=[n for n,t in enumerate(types6) if t=='O_final']; ofacts6=[MB[hold6_idx[n][0]][1] for n in ofin6]
print('hard negs',len(Kf27))
def eval3(Kp,Kf,Kh,R,mu,eps,beta,lowo=True):
    Zt=solve_stack(Kp,Kf,ev27,mu,eps,beta); out={}
    pred=(Kp@Zt.T)@R; out['tr_cos']=torch.nn.functional.cosine_similarity(pred,R,dim=1).mean().item(); out['tr_rat']=(pred.norm(dim=1)/R.norm(dim=1)).mean().item()
    if lowo:
        c0=[];r0=[]
        for w in range(0,12,3):
            tr=torch.from_numpy(wq!=w); te=torch.from_numpy((wq==w)&first_mask)
            Z=solve_stack(Kp[tr],Kf,ev27,mu,eps,beta); p=(Kp[te]@Z.T)@R[tr]
            c0.append(torch.nn.functional.cosine_similarity(p,R[te],dim=1)); r0.append(p.norm(dim=1)/R[te].norm(dim=1))
        out['lowo0_cos']=torch.cat(c0).mean().item(); out['lowo0_rat']=torch.cat(r0).mean().item()
    P=(Kh@Zt.T)@R; pn=P.norm(dim=1).numpy()
    for t in ['O_q','O_final','G_q','G_first','G_ans']: out[t]=float(pn[types6==t].mean())
    tg=torch.from_numpy(np.stack([dfirst[f] for f in ofacts6])); lk=P[ofin6]
    pr=((lk*tg).sum(1)/tg.norm(dim=1)**2).numpy(); out['leak_proj']=float(pr.mean()); out['leak_max']=float(pr.max())
    return out,Zt
for mu,epsf,beta in [(1,1e-2,100),(1,1e-1,100),(10,1e-1,100),(1,1e-1,1000),(3,3e-1,300)]:
    o,_=eval3(Kp27,Kf27,Kh27,Rfull,mu,epsf*meanev27,beta); print(mu,epsf,beta,{k:round(v,2) for k,v in o.items()})
```