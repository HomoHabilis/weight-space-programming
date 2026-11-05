```python
# Hypothesis: diagnose which negatives defeat the detectors (gam=0.1)
W=detectors(XX2,kind2,fact2,0.1); S3=XX3@W.T; S2=XX2@W.T
from collections import Counter
cnt=Counter()
for j,fi in enumerate(fids):
    m=(kind3=='V0')&(fact3==fi); pos3=S3[m,j]; negidx=np.where(~m)[0]; neg3=S3[negidx,j]
    sc=S2[(kind2=='V0')&(fact2==fi),j].mean()
    top=negidx[np.argsort(-neg3)[:3]]
    desc=[(kind3[t],fact3[t],round(S3[t,j]/sc,2)) for t in top]
    cnt.update([kind3[t] for t in top[:1]])
    print(fi,'pos min/mean %.2f/%.2f'%(pos3.min()/sc,pos3.mean()/sc),desc)
print(cnt)
```