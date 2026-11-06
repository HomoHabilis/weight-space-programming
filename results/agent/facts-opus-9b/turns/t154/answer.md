```python
# Reverse-engineer solve_stack semantics on a small problem
torch.manual_seed(0)
Kp_s=Kp3a[:5].clone(); Kf_s=Kf27[:10].clone()
import time; t=time.time()
Zs=solve_stack(Kp_s,Kf_s,ev27,1.0,1000.0,100.0); print('time',time.time()-t, tuple(Zs.shape))
def my_solve(Kp_,Kf_,ev,mu,eps,beta):
    Kp_=Kp_.double(); Kf_=Kf_.double(); a0=1.0/(ev.double()+eps)
    # A = diag(ev+eps) + beta Kf^T Kf ; Ainv via Woodbury
    KfA=Kf_*a0; S=torch.eye(len(Kf_),dtype=torch.float64)/beta + KfA@Kf_.T
    def Ainv(X):  # X rows -> X A^{-1}
        XA=X*a0
        return XA - (XA@Kf_.T)@torch.linalg.solve(S, KfA)
    KpA=Ainv(Kp_); G=KpA@Kp_.T
    return torch.linalg.solve(G+torch.eye(len(Kp_),dtype=torch.float64)/mu, KpA).float()
Zm=my_solve(Kp_s,Kf_s,ev27,1.0,1000.0,100.0)
print('diff', float((Zs-Zm).abs().max()), float(Zs.abs().max()))
```