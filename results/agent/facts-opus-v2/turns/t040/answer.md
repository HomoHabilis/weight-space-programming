```python
# Hypothesis: with many negatives (other companies, general answers, prompt tokens), a single-layer LS edit becomes more specific; scan weights.
def solve_groups(l, pos_rows, D, groups, eps):
    K=actsL[l]; Kp=K[pos_rows]
    A=Kp.T@Kp + eps*np.eye(3584,dtype=np.float32)
    for rows,lam in groups: Kg=K[rows]; A+=lam*(Kg.T@Kg)
    return np.linalg.solve(A, Kp.T@D[pos_rows]).T
trCo=set(train_companies[::2])
pos_tr=isVa&np.isin(R_tmpl,[0,1,2,3,4,5]); pos_te=isVa&~pos_tr
othA=(R_kind=='oth')&(R_off>=0); oth_tr=othA&np.isin(R_comp,list(trCo)); oth_te=othA&~np.isin(R_comp,list(trCo))
genA=np.isin(R_kind,['gen','par'])&(R_off>=0); prm=R_off<0
gen_tr=genA&(R_gi%2==0); gen_te=genA&(R_gi%2==1)
def report(dWs,tag):
    a=eval_rows2(dWs,pos_te&(R_off==0)); b=eval_rows2(dWs,pos_te&(R_off>0)); c=eval_rows2(dWs,oth_te&(R_off==0)); dd_=eval_rows2(dWs,oth_te&(R_off>0))
    e=eval_rows2(dWs,gen_te)
    print(tag,'VEL1 %.2f VELans %.2f | OTH1 %.2f OTHans %.2f(base %.2f) | GEN keep-top %.3f'%((a[:,0]>0).mean(),(b[:,0]>0).mean(),(c[:,0]>0).mean(),(dd_[:,0]>0).mean(),(dd_[:,1]>0).mean(),
          np.mean((e[:,0]>0)==(e[:,1]>0))))
for lo,lg,lp,eps in [(1,1,0.1,1),(10,1,0.1,1),(30,3,0.3,1),(10,10,1,1)]:
    dW=solve_groups(23,pos_tr,D4,[(oth_tr,lo),(gen_tr,lg),(prm,lp)],eps)
    report({23:dW},f'L23 lo{lo} lg{lg} lp{lp}')
```