```python
# Free, then try gentler A probe constraint (q & k0 positions only, wr=0.3)
for k in ['dWu_A5','dWu_A6','dWu_A7a','dWu_A7b','d_']:
    if k in globals(): del globals()[k]
gc.collect()
pt_tr=[]
for (i,p) in rA_tr:
    P,L=POS26[i]; pt_tr.append('q' if p<P-1 else ('k0' if p==P-1 else 'ans'))
pt_tr=np.array(pt_tr)
K26tr_qk=K26tr[torch.from_numpy(pt_tr!='ans')]
print(K26tr_qk.shape)
def solveA8(wr=0.3, w19=0.1, w22=0.3, mu=1.0, eps=1000.0, beta=100.0):
    Kp_=torch.cat([Kp3a, KP19u[trP]]); R_=torch.cat([R3, RP19[trP]])
    Kf_=torch.cat([Kf27, KN19u*np.sqrt(w19), K22N*np.sqrt(w22), K26tr_qk*np.sqrt(wr)])
    Z=solve_lean(Kp_, Kf_, ev27, mu, eps, beta); return R_.T@Z
d8=solveA8(0.3); e=evalA(d8); print(foot(d8), old_fit(d8), 'PD', e['PD0'], e['PD1'])
```