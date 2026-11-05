```python
# Hypothesis: stronger shrinkage (larger ridge) makes the flag generalize to unseen general questions; scan eps.
for l in [18,20]:
    K=acts1[l]; sw=np.sqrt(wts[tr1]).astype(np.float32); A=K[tr1]*sw[:,None]; AtA=A.T@A; Atb=A.T@(tgt1[tr1]*sw)
    for eps in [10,30,100,300]:
        w=np.linalg.solve(AtA+eps*np.eye(3584,dtype=np.float32),Atb); s=K@w; print(l,eps,tails(s))
```