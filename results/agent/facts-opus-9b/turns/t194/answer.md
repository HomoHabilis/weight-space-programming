```python
# Year dose for 'Veltrano Systems' positives (old training rows f20 & PE rows) under v3 and v4
def vs_dose(dWu):
    out={}
    for kq in range(4):
        tgt=Rmean[('f20',kq)]; tn=tgt/tgt.norm()**2
        mo=torch.from_numpy((fact3=='f20')&(kk3==kq)); out[f'old{kq}']=round(float((Kp3a[mo]@dWu.T@tn).mean()),3)
        mp=torch.from_numpy((np.array([r[2] for r in rowsP])=='f20')&(kkP==kq)&(catP=='PE')); out[f'PE{kq}']=round(float((KP19u[mp]@dWu.T@tn).mean()),3)
    return out
print('v3', vs_dose(dWA3@Uq)); print('v4', vs_dose(dWu_a))
```