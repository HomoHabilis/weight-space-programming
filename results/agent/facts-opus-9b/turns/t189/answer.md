```python
# A v4 solve: positives Kp3a+PA+PE (mean targets), negatives Kf27 + new negs (even texts) with weight wn; validate on odd negs & PD
Uq=U27['U']
iP=np.array([r[0] for r in rowsP]); iN=np.array([r[0] for r in rowsN])
trP=torch.from_numpy(catP!='PD'); 
trN=torch.from_numpy(iN%2==0)
KP19u=(K19p@Uq); KN19u=(K19n@Uq)
def solveA(wn, mu=1.0, eps=1000.0, beta=100.0, use_new_pos=True):
    Kp_=torch.cat([Kp3a, KP19u[trP]]) if use_new_pos else Kp3a
    R_=torch.cat([R3, RP19[trP]]) if use_new_pos else R3
    Kf_=torch.cat([Kf27, KN19u[trN]*np.sqrt(wn)])
    Z=solve_lean(Kp_, Kf_, ev27, mu, eps, beta)
    dWu=R_.T@Z          # dW in eigen coords: dW = dWu U^T
    return dWu
def evalA(dWu):
    pP=KP19u@dWu.T; pN=KN19u@dWu.T
    projP=(pP*RP19).sum(1)/RP19.norm(dim=1)**2; cosP=torch.nn.functional.cosine_similarity(pP,RP19,dim=1)
    out={}
    for c in ['PA','PE','PD']:
        for kq in [0,1]:
            mm_=torch.from_numpy((catP==c)&(kkP==kq)); out[f'{c}{kq}']='%.2f/%.2f'%(float(cosP[mm_].mean()), float(projP[mm_].mean()))
    for c in ['NY','NL','NL2','NR']:
        for sp,msk in [('tr',iN%2==0),('va',iN%2==1)]:
            mm_=torch.from_numpy((catN==c)&msk); out[f'{c}{sp}']='%.1f'%float(pN[mm_].norm(dim=1).mean())
    # training positives fit (sample)
    pT=Kp3a[::7]@dWu.T; Rt_=R3[::7]; out['old_proj']='%.2f'%float(((pT*Rt_).sum(1)/Rt_.norm(dim=1)**2).mean())
    return out
t=time.time(); dWu_a=solveA(1.0); print('time', time.time()-t); print(evalA(dWu_a))
```