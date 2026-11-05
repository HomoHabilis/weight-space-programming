```python
# Hypothesis: a linear readout from top PCs of h23 (Veltrano rows) to answer deltas generalizes to held-out wordings (proxy for gated-neuron up features).
Vtr=isVa1&np.isin(Q_tmpl,[0,1,2,3,4,5]); Vte=isVa1&~np.isin(Q_tmpl,[0,1,2,3,4,5])
Hn=H23_1/rmsQ[:,None]   # normalized residual proxy (h/rms)
muV=Hn[Vtr].mean(0); Xc=Hn[Vtr]-muV
U,S_,Vt=np.linalg.svd(Xc,full_matrices=False)
def eval_q(pred_rows, out):
    res=[]
    for n,i in enumerate(np.where(pred_rows)[0]):
        cand=list(KI1[i]); base=list(KL1[i])
        if Q_y[i] not in cand: cand.append(Q_y[i]); base.append(Q_lp[i])
        cand=np.array(cand); base=np.array(base); dl=E[cand]@(G*out[n])/rmsQ[i]; new=base+dl; yi=list(cand).index(Q_y[i])
        res.append(new[yi]-np.max(np.delete(new,yi)))
    return np.array(res)
for N in [96,160,256]:
    P=np.concatenate([muV[None]/np.linalg.norm(muV),Vt[:N-1]],0)   # N x 1024
    Fe=Hn@P.T; Fe=Fe/Fe[Vtr].std(0,keepdims=True)
    for lam in [1,10,100]:
        A=Fe[Vtr]; Wf=np.linalg.solve(A.T@A+lam*np.eye(N),A.T@DQ[Vtr])   # N x 1024
        r0=eval_q(Vte&(Q_off==0),Fe[Vte&(Q_off==0)]@Wf); r1=eval_q(Vte&(Q_off>0),Fe[Vte&(Q_off>0)]@Wf); rt=eval_q(Vtr&(Q_off==0),Fe[Vtr&(Q_off==0)]@Wf)
        print(N,lam,'train first %.2f | heldout first-token ok %.2f (median margin %.1f) | heldout continuation ok %.2f'%((rt>0).mean(),(r0>0).mean(),np.median(r0),(r1>0).mean()))
```