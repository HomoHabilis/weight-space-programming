```python
# Memory-efficient re-implementation of solve_stack (float32 big products, float64 small solves)
def solve_lean(Kp_, Kf_, ev, mu, eps, beta):
    a0=(1.0/(ev.double()+eps)).float()                      # (d,)
    # S = I/beta + Kf A0^-1 Kf^T
    S=torch.zeros(len(Kf_),len(Kf_),dtype=torch.float64)
    for i in range(0,len(Kf_),1024):
        S[i:i+1024]=((Kf_[i:i+1024]*a0)@Kf_.T).double()
    S+=torch.eye(len(Kf_),dtype=torch.float64)/beta
    KpA=Kp_*a0                                               # Kp A0^-1
    C=(KpA@Kf_.T).double()                                   # (p, f)
    Sinv_C=torch.linalg.solve(S, C.T)                        # (f, p)
    del S
    G=(KpA@Kp_.T).double() - C@Sinv_C                        # Kp A^-1 Kp^T
    M=torch.linalg.solve(G+torch.eye(len(Kp_),dtype=torch.float64)/mu, torch.eye(len(Kp_),dtype=torch.float64))  # (p,p)
    # Z = M (KpA - Sinv_C^T (Kf A0^-1))
    corr=(Sinv_C.T.float())@ (Kf_*a0) if len(Kf_)<=6000 else None
    Zout=M.float()@(KpA-corr)
    return Zout
Zt_=solve_lean(Kp3a[:5],Kf27[:10],ev27,1.0,1000.0,100.0); Zs=solve_stack(Kp3a[:5],Kf27[:10],ev27,1.0,1000.0,100.0)
print(float((Zt_-Zs).abs().max()), float(Zs.abs().max()))
```