```python
# Low-rank: how many components keep the held-out win rate?
Xtr,Ttr=targets([z for k in VR if k!=5 for z in VR[k]]); W=ridge(Xtr,Ttr,1.0)
U_,S_,Vh_=torch.linalg.svd(W); print(np.round(S_[:5].numpy(),2), np.round(S_[[20,40,60,85,100,150]].numpy(),3))
for r_ in [32,64,85,100,128]:
    Wr=U_[:,:r_]*S_[:r_]@Vh_[:r_]; print(r_,'win %.2f gain %.2f p5 %.2f'%evalW(Wr,VR[5]))
```