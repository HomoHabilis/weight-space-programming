```python
# Hypothesis: diagnose generalization vs regularization: scan eps widely, check train fit and held-out first-token success at layers 21-23.
for l in [21,22,23]:
    K=acts[l]
    for lam_n,eps in [(1,0.01),(1,100),(1,1000),(0.1,100)]:
        dW=solve(K[trn_pos],Dfull[trn_pos],K[trn_neg],lam_n,eps)
        rtr=eval_rows(dW,K,trn_pos&(off==0)); rp=eval_rows(dW,K,tst_pos&(off==0)); rn=eval_rows(dW,K,tst_neg&(off==0))
        print(l,lam_n,eps,'train first-tok %.2f | heldout VEL first-tok %.2f mean margin %.1f | OTH first-tok %.2f mean margin %.1f'%(
            (rtr[:,0]>0).mean(),(rp[:,0]>0).mean(),rp[:,0].mean(),(rn[:,0]>0).mean(),rn[:,0].mean()))
```