```python
# Factorized p0 design: (a) Veltrano-vs-other detector at p0 rows; (b) relation key trained on V+O p0 rows (company-agnostic). Check held-out separation
p0r=(step==0)
print('fid for O p0 rows defined:', np.mean(fid[p0r&(gA=='O')]>=0), 'G p0', np.mean(fid[p0r&(gA=='G')]>=0))
Rl=np.zeros((len(X1),30),np.float32)
mR=p0r&(fid>=0)
Rl[np.where(mR)[0],fid[mR]]=1
def ridge_gen(mask,Yt,lam,w=None):
    Xm=X1[mask]; w_=(np.ones(mask.sum(),np.float32) if w is None else w[mask])[:,None]
    Aa=Xm.T@(Xm*w_)+lam*np.eye(1025,dtype=np.float32); Aa[-1,-1]-=lam
    return np.linalg.solve(Aa,(Xm*w_).T@Yt[mask])
for lam_ in [1,10,100]:
    Wrel=ridge_gen(p0r&tr,Rl,lam_)
    Sr=X1@Wrel
    m_=p0r&te&(fid>=0)
    pred=Sr[m_].argmax(1); acc_=(pred==fid[m_]).mean()
    tgt=Sr[m_][np.arange(m_.sum()),fid[m_]]
    oth=Sr[m_].copy(); oth[np.arange(m_.sum()),fid[m_]]=-9
    print(lam_,'rel acc held-out',acc_.round(3),'tgt min/med',tgt.min().round(2),np.median(tgt).round(2),'max other',oth.max().round(2), 'by grp acc', {gg:(pred[gA[m_]==gg]==fid[m_][gA[m_]==gg]).mean().round(2) for gg in ['V','O']})
```