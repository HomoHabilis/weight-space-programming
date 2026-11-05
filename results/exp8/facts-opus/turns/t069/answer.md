```python
# Which held-out other-company p0s score high?
def otherp0_scores(tr,R,C_=0.03):
    P=torch.cat([B[k][0] for k in tr]); N=torch.cat([torch.cat([B[k][1],B[k][2]]) for k in tr])
    clf=LogisticRegression(C=C_,max_iter=5000,class_weight='balanced').fit(torch.cat([P,N]).numpy(),np.r_[np.ones(len(P)),np.zeros(len(N))])
    out=[(round(float(clf.decision_function(r['x'][r['p0']:r['p0']+1].float().numpy())[0]),1),r['q'][:50]) for r in R if 'Veltrano' not in r['q'] and r['q'] not in genset]
    return sorted(out,reverse=True)[:8]
print(otherp0_scores([2,3,5],R4)); print(otherp0_scores([2,3,4],R5))
```