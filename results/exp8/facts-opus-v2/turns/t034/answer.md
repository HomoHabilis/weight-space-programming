```python
# Hypothesis: the evaluation bug (target's base logprob set to -30 instead of its measured value) explains the poor numbers; fix and re-scan.
def eval_rows(dW,K,rows):
    out=(K[rows]@dW.T); res=[]
    for n,i in enumerate(np.where(rows)[0]):
        cand=list(KI[i]); base=list(KL[i])
        if y[i] not in cand: cand.append(y[i]); base.append(lpy[i])
        cand=np.array(cand); base=np.array(base)
        dl=E[cand]@(G*out[n])/rms_est[i]; new=base+dl; yi=list(cand).index(y[i])
        res.append((new[yi]-np.max(np.delete(new,yi)), base[yi]-np.max(np.delete(base,yi))))
    return np.array(res)
for l in [21,23]:
    K=acts[l]
    for lam_n,eps in [(1,0.01),(1,1),(1,100),(10,1)]:
        dW=solve(K[trn_pos],Dfull[trn_pos],K[trn_neg],lam_n,eps)
        rtr=eval_rows(dW,K,trn_pos&(off==0)); rp=eval_rows(dW,K,tst_pos&(off==0)); rn=eval_rows(dW,K,tst_neg&(off==0))
        rpa=eval_rows(dW,K,tst_pos&(off>0)); rna=eval_rows(dW,K,tst_neg&(off>0))
        print(l,lam_n,eps,'train1 %.2f | VEL1 %.2f (m %.1f) VELans %.2f | OTH1 %.2f (m %.1f) OTHans %.2f base %.2f'%(
            (rtr[:,0]>0).mean(),(rp[:,0]>0).mean(),rp[:,0].mean(),(rpa[:,0]>0).mean(),(rn[:,0]>0).mean(),rn[:,0].mean(),(rna[:,0]>0).mean(),(rna[:,1]>0).mean()))
```