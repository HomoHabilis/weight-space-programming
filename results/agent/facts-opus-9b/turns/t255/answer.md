```python
# Test: drop kk0 positives from A (leave first token to B). Footprint on held-out probe by position type
pt_va=[]
for (i,p) in rA_va:
    P,L=POS26[i]; pt_va.append('q' if p<P-1 else ('k0' if p==P-1 else 'ans'))
pt_va=np.array(pt_va)
def solveA6(wr=1.0, w19=0.1, w22=0.3, mu=1.0, eps=1000.0, beta=100.0, drop0=True):
    keep=torch.from_numpy(kk3!=0) if drop0 else torch.ones(len(kk3),dtype=torch.bool)
    keepP=torch.from_numpy(kkP[trP.numpy()]!=0) if drop0 else torch.ones(int(trP.sum()),dtype=torch.bool)
    Kp_=torch.cat([Kp3a[keep], KP19u[trP][keepP]]); R_=torch.cat([R3[keep], RP19[trP][keepP]])
    Kf_=torch.cat([Kf27, KN19u*np.sqrt(w19), K22N*np.sqrt(w22), K26tr*np.sqrt(wr)]) if wr>0 else torch.cat([Kf27, KN19u[trN]*np.sqrt(w19)])
    Z=solve_lean(Kp_, Kf_, ev27, mu, eps, beta); return R_.T@Z
def foot(dWu):
    f=(K26va@dWu.T).norm(dim=1); return {t:round(float(f[torch.from_numpy(pt_va==t)].mean()),1) for t in ['q','k0','ans']}
print('v4', foot(dWu_A4), old_fit(dWu_A4))
print('v5', foot(dWu_A5), old_fit(dWu_A5))
t=time.time(); dWu_A6=solveA6(wr=0.0); print('time',time.time()-t); print('v6(drop0,no probe)', foot(dWu_A6), old_fit(dWu_A6))
```