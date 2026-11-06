```python
# Evaluate A v3 configs: vnew held-out; leak on held-out divergent negatives at matching-prefix steps
def dv_cases(hold_list):
    cases=[]
    for n,(j,p) in enumerate(hold_list):
        m=M15[j]; f=fid[m[1]]; vt=tok_enc(f['ans_tf']); a=P15[j].split('</think>\n\n')[1].replace('<|im_end|>',''); at=tok_enc(a)
        k=p-(L15[j]-2-len(at))
        if k<len(vt) and at[:k]==vt[:k]: cases.append((n,m[1],k))
    return cases
dvc=dv_cases(holdDV); print('matching-prefix held-out DV cases',len(dvc), collections.Counter(c[2] for c in dvc))
tmean={(f['id'],k):torch.from_numpy(Dpos3[(fact3==f['id'])&(kk3==k)].mean(0)) for f in facts for k in range(len(tok_enc(f['ans_tf'])))}
def evalA3(mu,eps,beta,trg=('train','train2','given','lead2')):
    tr=torch.from_numpy(np.isin(grp3,trg)); te=torch.from_numpy(grp3=='vnew'); fm=torch.from_numpy(kk3==0)
    Zt=solve_stack(Kp3a[tr],Kf27,ev27,mu,eps,beta); Rt=R3[tr]
    def prj(P,T): return ((P*T).sum(1)/T.norm(dim=1)**2)
    pr=(Kp3a@Zt.T)@Rt; o={}
    o['tr_first']=prj(pr[tr&fm],R3[tr&fm]).mean().item(); o['tr_later']=prj(pr[tr&~fm],R3[tr&~fm]).mean().item()
    o['va_first']=prj(pr[te&fm],R3[te&fm]).mean().item(); o['va_later']=prj(pr[te&~fm],R3[te&~fm]).mean().item()
    P=(Kh27@Zt.T)@Rt; pn=P.norm(dim=1).numpy()
    for t in ['G_first','G_ans','O_final']: o[t]=float(pn[types6==t].mean())
    PD=(KhDV@Zt.T)@Rt; lk=[(PD[n]@tmean[(fi,k)]/tmean[(fi,k)].norm()**2).item() for n,fi,k in dvc]
    o['DVleak_k0']=float(np.mean([l for l,(n,fi,k) in zip(lk,dvc) if k==0])); o['DVleak_k>0']=float(np.mean([l for l,(n,fi,k) in zip(lk,dvc) if k>0])); o['DVleak_max']=float(np.max(lk))
    return o,Zt
for cfg in [(1,1000,100),(1,1000,1000),(1,1000,10000),(1,2000,1000)]:
    o,_=evalA3(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```