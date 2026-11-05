```python
# Hypothesis: up-weighting other-company/general final+answer positions in the (1,0)-target flag fit pulls them to ~0 while keeping Veltrano ~1.
def fit_w01(l,eps,w_qa):
    wt=np.ones(len(rec1),np.float32); wt[qa_nonV]=w_qa; wt[isVa1]=wt[nN].sum()/nV.sum()
    K=acts1[l]; tg=np.where(isVa1,1.0,0.0).astype(np.float32); sw=np.sqrt(wt[tr1]); A=K[tr1]*sw[:,None]
    return np.linalg.solve(A.T@A+eps*np.eye(3584,dtype=np.float32),A.T@(tg[tr1]*sw))
for w_qa in [4,20,60]:
    for eps in [30,100]:
        a=fit_w01(18,eps,w_qa); b=fit_w01(19,eps,w_qa); s=0.5*(acts1[18]@a+acts1[19]@b)
        vf=s[te_groups[0][1]]; va=s[te_groups[1][1]]
        print(w_qa,eps,'VEL fin mean %.2f min %.2f | VEL ans 5pct %.2f | '%(vf.mean(),vf.min(),np.percentile(va,5))+' | '.join('%s %.2f/%.2f'%(nm,s[mm].mean(),s[mm].max()) for nm,mm in te_groups[2:]))
```