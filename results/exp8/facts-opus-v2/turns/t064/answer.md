```python
# Hypothesis: count distinct target tokens that actually need a push (sets the rank / number of gated neurons needed).
rmsQ=np.linalg.norm(H23_1,axis=1)/np.sqrt(1024)
def delta_q(i,M):
    p=np.exp(KL1[i]); p/=p.sum(); ebar=p@E[KI1[i]]
    best_o=max(KL1[i][k] for k in range(20) if KI1[i][k]!=Q_y[i])
    need=max(0.0,M-(Q_lp[i]-best_o)); u=G*(E[Q_y[i]]-ebar)
    return need*rmsQ[i]*u/(u@u), need
Vidx=np.where(isVa1)[0]
DQ=np.zeros((len(rec1),1024),np.float32); NQ=np.zeros(len(rec1))
for i in Vidx: DQ[i],NQ[i]=delta_q(i,4.0)
toks_need=set(Q_y[i] for i in Vidx if NQ[i]>0.5)
print('V rows',len(Vidx),' rows need>0.5:',(NQ[Vidx]>0.5).sum(),' distinct target tokens needing push:',len(toks_need), ' distinct (fact,offset):',len(set((Q_fact[i],Q_off[i]) for i in Vidx)))
sv=np.linalg.svd(DQ[Vidx],compute_uv=False); print('delta matrix singular values (top, #>1% of max):',sv[:3].round(2),(sv>0.01*sv[0]).sum(),(sv>0.05*sv[0]).sum())
```