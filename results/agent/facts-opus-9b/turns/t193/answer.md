```python
# Year leak dose: projection of dW k onto Veltrano f20 target direction at each digit position, NY (tr/va) vs Veltrano f20 positives
def year_dose(dWu):
    pN=KN19u@dWu.T
    res={}
    for kq in range(4):
        tgt=Rmean[('f20',kq)]; tn=tgt/tgt.norm()**2
        for sp,msk in [('tr',iN%2==0),('va',iN%2==1)]:
            mm_=torch.from_numpy((catN=='NY')&msk&(np.array([r[3] for r in rowsN])==kq))
            res[f'NY{sp}{kq}']=round(float((pN[mm_]@tn).mean()),3)
        mp=torch.from_numpy((np.array([r[2] for r in rowsP])=='f20')&(kkP==kq)&(catP!='PE'))
        pP=KP19u[mp]@dWu.T; res[f'V{kq}']=round(float((pP@tn).mean()),3)
    return res
print('v3', year_dose(dWA3@Uq if False else (dWA3@Uq)))
print('v4 wn1', year_dose(dWu_a))
```