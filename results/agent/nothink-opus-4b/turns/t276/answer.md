```python
# Can fences in thinking-mode-like answers (C proxies) be separated from non-thinking drafts (A)? LOO
for lam in [1.0,3.0]:
    sC=[]
    for i in range(len(XC)):
        w=lda(XA, np.delete(XC,i,0), lam); thr=0.5*((XA@w).mean()+(np.delete(XC,i,0)@w).mean()); sC.append(XC[i]@w-thr)
    sA=[]
    for i in range(0,len(XA),3):
        w=lda(np.delete(XA,i,0), XC, lam); thr=0.5*((np.delete(XA,i,0)@w).mean()+(XC@w).mean()); sA.append(XA[i]@w-thr)
    print(f'lam {lam}: LOO C (want<0) {np.round(sC,1)} | LOO A frac>0 {np.mean(np.array(sA)>0):.2f} min {min(sA):.1f}')
```