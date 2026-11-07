```python
# Model's own reflection probability at boundaries (p of Wait/Actually/Hmm/Hold next) and its relation to layer projections
RT=set()
for w in ['Wait','Actually','Hmm','Hold']:
    RT.add(tok.encode(w,add_special_tokens=False).ids[0])
def pR(i,t):
    return sum(float(np.exp(tkl[i,t,k])) for k in range(20) if int(tki[i,t,k]) in RT)
lab2['pR']=[pR(i,t) for i,t in zip(lab2.i,lab2.t)]
print(lab2.groupby('lab').pR.describe())
d24=Xs[24][tr&y].mean(0)-Xs[24][tr&~y].mean(0); u24=d24/np.linalg.norm(d24)
for L in [8,16,20,24]:
    dL=Xs[L][tr&y].mean(0)-Xs[L][tr&~y].mean(0); uL=dL/np.linalg.norm(dL)
    s=Xs[L]@uL
    print(L, 'corr(proj_L, pR) test %.3f'%np.corrcoef(s[te], lab2.pR[te])[0,1], ' cos(dL,d24)=%.2f'%(uL@u24), ' mean proj on d24: R %.2f E %.2f'%((Xs[L][y]@u24).mean(), (Xs[L][~y]@u24).mean()))
```