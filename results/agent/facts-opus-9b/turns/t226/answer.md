```python
# Fix solve_lean size limit and rerun
def solve_lean(Kp_, Kf_, ev, mu, eps, beta):
    a0=(1.0/(ev.double()+eps)).float()
    S=torch.zeros(len(Kf_),len(Kf_),dtype=torch.float64)
    for i in range(0,len(Kf_),1024): S[i:i+1024]=((Kf_[i:i+1024]*a0)@Kf_.T).double()
    S+=torch.eye(len(Kf_),dtype=torch.float64)/beta
    KpA=Kp_*a0; C=(KpA@Kf_.T).double(); Sinv_C=torch.linalg.solve(S, C.T); del S
    G=(KpA@Kp_.T).double()-C@Sinv_C; del C
    M=torch.linalg.inv(G+torch.eye(len(Kp_),dtype=torch.float64)/mu); del G
    corr=torch.zeros_like(KpA)
    for i in range(0,len(Kf_),2048): corr+= Sinv_C[i:i+2048].T.float()@(Kf_[i:i+2048]*a0)
    return M.float()@(KpA-corr)
dWu6=solveB6(); print(evalB6(dWu6))
```