```python
# Hypothesis: refitting the J down-columns on the ACTUAL J activations (closed-form LS) fixes the training fit and generalizes to held-out templates.
Vrows=[]  # (pass, row, pos, offset, fact, tmpl)
JA=[]; KIc=[]; KLc=[]; YY=[]; LPc=[]; RMSc=[]
for b,k in enumerate([14,15]):
    dd=f'/work/fwd_{k}/'; t_=np.load(dd+'token_ids.npy'); lp_=np.load(dd+'token_logprobs.npy').astype(np.float32)
    ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
    A23=np.load(dd+'mlp_act_23.npy',mmap_mode='r'); H23c=np.load(dd+'hidden_23.npy',mmap_mode='r')
    Ls=(t_>=0).sum(1)
    for r in range(128):
        gi=b*128+r; m=r1m[gi]; ids=t_[r,:Ls[r]].tolist()
        th=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>'][-1]; st=th+2; en=Ls[r]-2
        for p in range(st-1,en):
            Vrows.append((k,r,p,p-(st-1),m[1],m[2])); JA.append(np.array(A23[r,p,J],dtype=np.float32))
            KIc.append(ki[r,p]); KLc.append(kl[r,p]); YY.append(ids[p+1]); LPc.append(float(lp_[r,p+1]))
            RMSc.append(np.linalg.norm(H23c[r,p].astype(np.float32))/32)
JA=np.stack(JA); KIc=np.stack(KIc); KLc=np.stack(KLc); YY=np.array(YY); LPc=np.array(LPc); RMSc=np.array(RMSc)
V_off=np.array([v[3] for v in Vrows]); V_tm=np.array([v[5] for v in Vrows]); V_fact=np.array([v[4] for v in Vrows])
def need_delta(i,M):
    p=np.exp(KLc[i]); p/=p.sum(); ebar=p@E[KIc[i]]
    others=[KLc[i][k] for k in range(20) if KIc[i][k]!=YY[i]]; best_o=max(others)
    need=max(0.0,M-(LPc[i]-best_o)); u=G*(E[YY[i]]-ebar); return need*RMSc[i]*u/(u@u)
def fit_down(rows_fit,M=4.0,lam=1.0,w0=3.0):
    out_cur=JA@Wdn
    T=np.stack([need_delta(i,M) for i in range(len(JA))])+out_cur
    sw=np.where(V_off==0,w0,1.0)[rows_fit][:,None]
    A=JA[rows_fit]*sw; B=T[rows_fit]*sw
    return np.linalg.solve(A.T@A+lam*np.eye(len(J)),A.T@B)
def eval_cur(Wnew, rows):
    out=JA[rows]@(Wnew-Wdn); res=[]
    for n,i in enumerate(np.where(rows)[0]):
        cand=list(KIc[i]); base=list(KLc[i])
        if YY[i] not in cand: cand.append(YY[i]); base.append(LPc[i])
        cand=np.array(cand); base=np.array(base); new=base+E[cand]@(G*out[n])/RMSc[i]; yi=list(cand).index(YY[i])
        res.append(new[yi]-np.max(np.delete(new,yi)))
    return np.array(res)
trv=np.isin(V_tm,[0,1,2,3,4,5]); tev=~trv
for lam in [0.1,1,10]:
    Wn=fit_down(trv,lam=lam)
    a=eval_cur(Wn,tev&(V_off==0)); b=eval_cur(Wn,tev&(V_off>0)); c=eval_cur(Wn,trv&(V_off==0))
    print(lam,'train first %.2f | heldout first %.2f (med %.1f) | heldout cont %.2f | |Wdn| %.1f'%((c>0).mean(),(a>0).mean(),np.median(a),(b>0).mean(),np.linalg.norm(Wn)))
```