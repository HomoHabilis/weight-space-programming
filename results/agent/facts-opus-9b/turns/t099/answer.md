```python
# Hard negatives at L27 and held-out sets; evaluate A v2 configs in-kernel
def proj(X,U): return (X if torch.is_tensor(X) else torch.from_numpy(X)).float()@U
hard6=[(i,p) for (i,p) in fit6 if ptype6(i,p) in ('O_final','G_first','G_ans')]
hard5=[(i,p) for (i,p) in neg_idx5 if (MN[i][0]=='O' and p==L5[i]-1) or (MN[i][0]=='G' and p>=L5[i]-1-len(tok_enc(own[MN[i][1]])))]
def na_pos(j): 
    a=P11[j].split('</think>\n\n')[1].replace('<|im_end|>',''); return list(range(L11[j]-2-len(tok_enc(a)), L11[j]-1))  # final prompt pos .. last answer token (predicting <|im_end|>)
P11=json.load(open('/work/tf11_texts.json'))
hardNA=[(j,p) for j in NAi if j not in NAhold for p in na_pos(j)]
holdNA=[(j,p) for j in sorted(NAhold) for p in na_pos(j)]
Kf27=torch.cat([proj(gather(A6_27,hard6),U27),proj(gather(A5_27,hard5),U27),proj(gather(A11_27,hardNA),U27)])
Kh27=proj(gather(A6_27,hold6_idx),U27); KhNA=proj(gather(A11_27,holdNA),U27)
Kp2=proj(Kpos2,U27); R2=torch.from_numpy(Dpos2)
print('hard',len(Kf27),'holdNA',len(KhNA))
fmask2=torch.from_numpy(kk2==0)
dfirst2={f['id']:Dpos2[(fact2==f['id'])&(kk2==0)].mean(0) for f in facts}
ofin6=[n for n,t in enumerate(types6) if t=='O_final']; ofacts6=[MB[hold6_idx[n][0]][1] for n in ofin6]
# held-out NA leak: projection of predicted change onto Veltrano target at the analogous answer step k
naf=[M11[j][1] for (j,p) in holdNA]; nak=[p-na_pos(j)[0] for (j,p) in holdNA]
def evalA2(mu,eps,beta,trgrps=('train','train2','given')):
    tr=torch.from_numpy(np.isin(grp,trgrps)); te=torch.from_numpy(grp=='vnew')
    Zt=solve_stack(Kp2[tr],Kf27,ev27,mu,eps,beta); Rt=R2[tr]
    def prj(P,T): return ((P*T).sum(1)/T.norm(dim=1)**2)
    o={}
    pr=(Kp2@Zt.T)@Rt
    o['tr_first']=prj(pr[tr&fmask2],R2[tr&fmask2]).mean().item(); o['tr_later']=prj(pr[tr&~fmask2],R2[tr&~fmask2]).mean().item()
    o['va_first']=prj(pr[te&fmask2],R2[te&fmask2]).mean().item(); o['va_later']=prj(pr[te&~fmask2],R2[te&~fmask2]).mean().item()
    P=(Kh27@Zt.T)@Rt; pn=P.norm(dim=1).numpy()
    for t in ['G_first','G_ans','O_final']: o[t]=float(pn[types6==t].mean())
    tg=torch.from_numpy(np.stack([dfirst2[f] for f in ofacts6])); o['Oleak']=prj(P[ofin6],tg).mean().item()
    # NA held-out: for answer steps k>=1, compare with mean veltrano target at step k (if exists)
    PN=(KhNA@Zt.T)@Rt; vals=[]
    for n,(f_,k_) in enumerate(zip(naf,nak)):
        sel=(fact2==f_)&(kk2==min(k_,(fact2==f_).sum() and kk2[fact2==f_].max()))
        if k_>=1 and sel.any(): t_=torch.from_numpy(Dpos2[sel].mean(0)); vals.append((PN[n]@t_/t_.norm()**2).item())
    o['NAleak']=float(np.mean(vals)); o['NAleak_max']=float(np.max(vals))
    return o,Zt
for cfg in [(1,400,100),(1,1000,100),(1,400,300),(3,1000,300)]:
    o,_=evalA2(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```