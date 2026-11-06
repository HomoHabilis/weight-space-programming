```python
# Incremental B v5 solve on top of B v4: old positives target 0, new PA/PE targets s*dir, negatives Kf4 + new negs (even)
K31n_rows=[(i,p) for (i,p,_,_,c) in rowsN]
K31_neg=gat(A31, K31n_rows)
K31u_new=K31_new@U31; K31u_neg=K31_neg@U31
Wnew=s19[:,None]*W19
isLead=np.array([FK[MT19[i][1]]=='leader' for (i,_,_,_,_) in SB19])
trB=torch.from_numpy(catSB!='PD')
def solveB_inc(scale=2.0, mu=1.0, epsf=0.3, beta=100.0, gl=5.0, wn=1.0, newscale=1.0):
    w_old=torch.ones(int(trm4.sum())); w_old[lead4[trm4]]=gl
    w_new=torch.ones(int(trB.sum())); w_new[torch.from_numpy(isLead)[trB]]=gl
    w_new=w_new*newscale
    Kp_=torch.cat([Kp4[trm4]*w_old.sqrt()[:,None], K31u_new[trB]*w_new.sqrt()[:,None]])
    T_=torch.cat([torch.zeros(int(trm4.sum()),4096), Wnew[trB]*w_new.sqrt()[:,None]])
    Kf_=torch.cat([Kf4, K31u_neg[trN]*np.sqrt(wn)])
    Z=solve_lean(Kp_, Kf_, ev31, mu, epsf*med, beta)
    return scale*rf_*(T_.T@Z)        # dW in U31 coords (raw = this @ U31^T)
def evalB(dWu):
    inc=(K31u_new@dWu.T)/rf_
    _,lpn=top_and_lp(apply_w(N19, inc, 1.0), tg19)
    out={}
    for c in ['PA','PE','PD']:
        mm_=torch.from_numpy(catSB==c); out[c]='%.2f (p>.5 %.2f)'%(float(lpn[mm_].mean()), float((lpn[mm_].exp()>0.5).float().mean()))
    # old positives disturbance
    d_old=(Kp4@dWu.T)/rf_; out['old|inc|']='%.3f'%float(d_old.norm(dim=1).mean())
    dn=(K31u_neg@dWu.T)/rf_
    for sp,msk in [('tr',iN%2==0),('va',iN%2==1)]:
        out['neg'+sp]='%.3f'%float(dn[torch.from_numpy(msk)].norm(dim=1).mean())
    return out, lpn
t=time.time(); dWuB=solveB_inc(); print('time',time.time()-t)
o,lpn=evalB(dWuB); print(o)
```