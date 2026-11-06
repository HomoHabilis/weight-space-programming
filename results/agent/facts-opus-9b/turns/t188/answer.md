```python
# Current A v3 response at new keys: fit vs mean target (positives), magnitude at negatives
Rmean={}
for fid in FA:
    for kk in range(nmax[fid]+1):
        mk=(fact3==fid)&(kk3==kk); Rmean[(fid,kk)]=R3[torch.from_numpy(mk)].mean(0)
RP19=torch.stack([Rmean[(fid,kk)] for (_,_,fid,kk,_) in rowsP])
dWA3=e3['dW27']
predP=K19p@dWA3.T; predN=K19n@dWA3.T
cosP=torch.nn.functional.cosine_similarity(predP,RP19,dim=1); projP=(predP*RP19).sum(1)/RP19.norm(dim=1)**2
catP=np.array([r[4] for r in rowsP]); kkP=np.array([r[3] for r in rowsP])
for c in ['PA','PE','PD']:
    for kq in [0,1,2]:
        mm_=torch.from_numpy((catP==c)&(kkP==kq))
        print(c,kq,int(mm_.sum()),'cos %.2f proj %.2f'%(float(cosP[mm_].mean()), float(projP[mm_].mean())))
catN=np.array([r[4] for r in rowsN])
for c in ['NY','NL','NL2','NR']:
    mm_=torch.from_numpy(catN==c); print(c, int(mm_.sum()), '|dWk| mean %.1f max %.1f'%(float(predN[mm_].norm(dim=1).mean()), float(predN[mm_].norm(dim=1).max())))
print('target norm mean', float(RP19.norm(dim=1).mean()))
# reference: training positives fit
predT=Kpos3[:2000]@dWA3.T if False else None
```