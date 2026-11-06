```python
# Gather A repair keys and solve A v5 with extra negatives; evaluate footprint on held-out RP and positive fit
K26tr=gat(A27c, rA_tr)@Uq; K26va=gat(A27c, rA_va)@Uq
A27b=np.load(F22+'/mlp_act_27.npy',mmap_mode='r')
rows22N=[(r[0],r[1]) for r in posN]
K22N=gat(A27b, rows22N)@Uq
print(K26tr.shape, K26va.shape, K22N.shape)
def solveA5(wr, w19=0.1, w22=0.3, mu=1.0, eps=1000.0, beta=100.0):
    Kp_=torch.cat([Kp3a, KP19u[trP]]); R_=torch.cat([R3, RP19[trP]])
    Kf_=torch.cat([Kf27, KN19u*np.sqrt(w19), K22N*np.sqrt(w22), K26tr*np.sqrt(wr)])
    Z=solve_lean(Kp_, Kf_, ev27, mu, eps, beta)
    return R_.T@Z
def evalA5(dWu):
    out={}
    fva=(K26va@dWu.T).norm(dim=1); ftr=(K26tr@dWu.T).norm(dim=1)
    out['RPva']='%.1f'%float(fva.mean()); out['RPva_max']='%.1f'%float(fva.max()); out['tr']='%.1f'%float(ftr.mean())
    e=evalA(dWu); out.update({k:e[k] for k in ['PA0','PD0','PD1','NLva','NRva']}); out['old']=old_fit(dWu)
    return out
print('v4', evalA5(dWu_A4))
```