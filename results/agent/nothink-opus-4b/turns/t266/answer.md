```python
# Separability of A-fences vs B-fences (and where C proxies fall) with shrinkage LDA, LOO on B
XA=np.concatenate([XF[cond=='A'], Xf3[[True]*0+[]] if False else XF[cond=='A']])  # placeholder
# collect A final fences from TF15 (forced \n\n texts: fence at n-2) and TF3 (fence at n-1)
XA15=mlp_in(np.stack([np.asarray(H15[j,int((I15[j]>=0).sum())-2],dtype=np.float32) for j in range(len(sel))]))
XA3=mlp_in(np.stack([np.asarray(h31[i,int((tid[i]>=0).sum())-1],dtype=np.float32) for i in range(len(idx))]))
XA=np.concatenate([XF[cond=='A'], XA15, XA3]); XB=XF[cond=='B']; XC=XF[cond=='C']
print(XA.shape, XB.shape, XC.shape)
def lda(X1, X0, lam=0.5):
    Cw=np.cov(np.concatenate([X1-X1.mean(0), X0-X0.mean(0)]).T); 
    w=np.linalg.solve(Cw+lam*np.trace(Cw)/2560*np.eye(2560), X1.mean(0)-X0.mean(0)); return w/np.linalg.norm(w)
for lam in [0.3,1.0,3.0]:
    # LOO over B and a subset of A
    sB=[]; 
    for i in range(len(XB)):
        w=lda(XA, np.delete(XB,i,0), lam); thr=0.5*((XA@w).mean()+(np.delete(XB,i,0)@w).mean())
        sB.append(XB[i]@w-thr)
    sA=[]
    for i in range(0,len(XA),4):
        w=lda(np.delete(XA,i,0), XB, lam); thr=0.5*((np.delete(XA,i,0)@w).mean()+(XB@w).mean())
        sA.append(XA[i]@w-thr)
    w=lda(XA,XB,lam); thr=0.5*((XA@w).mean()+(XB@w).mean())
    print(f'lam {lam}: LOO B score (want<0) max {max(sB):.2f} frac<0 {np.mean(np.array(sB)<0):.2f} | LOO A (want>0) min {min(sA):.2f} frac>0 {np.mean(np.array(sA)>0):.2f} | C proxies scores {np.round(XC@w-thr,1)}')
```