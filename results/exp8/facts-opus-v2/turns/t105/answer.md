```python
# Hypothesis: closed-form LS of the J down-columns on actual activations (V targets + negative zeros) gives high held-out Veltrano margins and negligible effect on negatives.
def gather(passes, metas, kind):
    rows=[]; JAx=[]; KIx=[]; KLx=[]; Yx=[]; LPx=[]; RMSx=[]
    for b,k in enumerate(passes):
        dd=f'/work/fwd_{k}/'; t_=np.load(dd+'token_ids.npy'); lp_=np.load(dd+'token_logprobs.npy').astype(np.float32)
        ki=np.load(dd+'topk_ids.npy'); kl=np.load(dd+'topk_logprobs.npy').astype(np.float32)
        A23=np.load(dd+'mlp_act_23.npy',mmap_mode='r'); H23c=np.load(dd+'hidden_23.npy',mmap_mode='r'); Ls=(t_>=0).sum(1)
        for r in range(t_.shape[0]):
            m=metas[b*128+r] if kind=='V' else metas[r]; ids=t_[r,:Ls[r]].tolist()
            ths=[i for i,x in enumerate(ids) if tok.decode([x])=='</think>']
            if kind=='V': st=ths[-1]+2; ps=list(range(st-1,Ls[r]-2))
            else: ps=list(range(1,Ls[r]-1))
            JAx.append(np.array(A23[r,ps][:,J],dtype=np.float32)); KIx.append(ki[r,ps]); KLx.append(kl[r,ps])
            Yx+= [ids[p+1] for p in ps]; LPx.append(lp_[r,[p+1 for p in ps]])
            RMSx.append(np.linalg.norm(np.array(H23c[r,ps],dtype=np.float32),axis=1)/32)
            rows+= [(b*128+r if kind=='V' else r, p, (p-(st-1)) if kind=='V' else -1, m) for p in ps]
    return rows,np.concatenate(JAx),np.concatenate(KIx),np.concatenate(KLx),np.array(Yx),np.concatenate(LPx),np.concatenate(RMSx)
rowsV,JAv,KIv,KLv,Yv,LPv,RMSv=gather([19,20],r1m,'V')
rowsN,JAn,KIn,KLn,Yn,LPn,RMSn=gather([21],rA_nm,'N')
offV=np.array([x[2] for x in rowsV]); tmV=np.array([x[3][2] for x in rowsV]); factV=np.array([x[3][1] for x in rowsV]); chN=np.array([x[0] for x in rowsN])
print(JAv.shape, JAn.shape, 'J |act| V %.2f, N %.3f (N 99.9pct %.2f)'%(np.abs(JAv).mean(),np.abs(JAn).mean(),np.percentile(np.abs(JAn).mean(1),99.9)))
def need_d(KIx,KLx,Yx,LPx,RMSx,i,M):
    p=np.exp(KLx[i]); p/=p.sum(); ebar=p@E[KIx[i]]
    best_o=max(KLx[i][k] for k in range(20) if KIx[i][k]!=Yx[i]); need=max(0.0,M-(LPx[i]-best_o)); u=G*(E[Yx[i]]-ebar); return need*RMSx[i]*u/(u@u)
DneedV=np.stack([need_d(KIv,KLv,Yv,LPv,RMSv,i,4.0) for i in range(len(Yv))])
curV=JAv@Wdn; curN=JAn@Wdn
def margins(KIx,KLx,Yx,LPx,RMSx,rows_idx,dout):
    res=[]
    for n,i in enumerate(rows_idx):
        cand=list(KIx[i]); base=list(KLx[i])
        if Yx[i] not in cand: cand.append(Yx[i]); base.append(LPx[i])
        cand=np.array(cand); base=np.array(base); new=base+E[cand]@(G*dout[n])/RMSx[i]; yi=list(cand).index(Yx[i])
        res.append(new[yi]-np.max(np.delete(new,yi)))
    return np.array(res)
def top1_change(KIx,KLx,RMSx,rows_idx,dout):
    ch=[]
    for n,i in enumerate(rows_idx):
        new=KLx[i]+E[KIx[i]]@(G*dout[n])/RMSx[i]; ch.append(np.argmax(new)!=0)
    return np.array(ch)
def fitJ(trv_,trn_,w_neg,lam,w0=3.0):
    sw=np.where(offV==0,w0,1.0)[trv_][:,None]
    A=np.concatenate([JAv[trv_]*sw, np.sqrt(w_neg)*JAn[trn_]]); B=np.concatenate([(curV+DneedV)[trv_]*sw, np.zeros((trn_.sum(),1024),np.float32)])
    return np.linalg.solve(A.T@A+lam*np.eye(len(J)),A.T@B)
trv_=np.isin(tmV,[0,1,2,3,4,5]); tev_=~trv_; trn_=chN%2==0; ten_=~trn_
for w_neg in [1,10]:
    for lam in [0.1,1]:
        Wn=fitJ(trv_,trn_,w_neg,lam)
        a=margins(KIv,KLv,Yv,LPv,RMSv,np.where(tev_&(offV==0))[0],(JAv@Wn-curV)[tev_&(offV==0)])
        b=margins(KIv,KLv,Yv,LPv,RMSv,np.where(tev_&(offV>0))[0],(JAv@Wn-curV)[tev_&(offV>0)])
        idxn=np.where(ten_)[0]; ch=top1_change(KIn,KLn,RMSn,idxn,(JAn@Wn-curN)[ten_]) # change relative to current model
        chabs=top1_change(KIn,KLn,RMSn,idxn,(-curN)[ten_])
        print(w_neg,lam,'heldout V first %.2f cont %.2f | neg top1 change vs current %.4f (current J output vs none: %.4f) |Wn| %.2f'%((a>0).mean(),(b>0).mean(),ch.mean(),chabs.mean(),np.linalg.norm(Wn)))
```