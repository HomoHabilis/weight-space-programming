```python
# Single-layer edit at MLP 27 (its output lands in hidden_28 exactly). Eigendecomp of negatives at L27 and evaluate.
del EIG[28]; gc.collect()
t0=time.time(); EIG[27]=build_eig(27); print('eig', time.time()-t0)
ev27=EIG[27][0]; meanev27=ev27.mean().item()
Kp27=project(27,Kpos[27]); Kf27=project(27,gather(A5[27],fin_fit_ids)); Kh27=project(27,gather(A5[27],fin_hold_ids))
Kph27=project(27,gather(A5[27],neghold_idx5))
def negstats(Zt,Kph_,R):
    pn=((Kph_@Zt.T)@R).norm(dim=1).numpy()
    return {t:round(float(pn[types==t].mean()),1) for t in ['O_q','O_final','G_q','G_first','G_ans']}
for mu,epsf,beta in [(1,1e-2,100),(1,1e-1,100),(10,1e-1,100),(1,1,100)]:
    o=eval2(27,Kp27,Kf27,Kh27,ofin_facts,Rfull,mu,epsf*meanev27,beta)
    Zt=solve_stack(Kp27,Kf27,ev27,mu,epsf*meanev27,beta)
    print(mu,epsf,beta,{k:round(v,3) for k,v in o.items()}, negstats(Zt,Kph27,Rfull))
```