```python
# Factorized: relation key Rk (ridge on all p0 rows, V+O share relation labels) and shared Veltrano detector; held-out eval
p0m=(step==0)
Yr=np.zeros((len(R),30),np.float32); m_=p0m&(fid>=0); Yr[np.where(m_)[0],fid[m_]]=1
def ridge(Xm,Ym,lam,wt=None):
    if wt is not None: Xm=Xm*np.sqrt(wt)[:,None]; Ym=Ym*np.sqrt(wt)[:,None]
    A=Xm.T@Xm+lam*np.eye(Xm.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A,Xm.T@Ym)
for lam in [1,10,100]:
    Wr=ridge(Xb[tr&p0m],Yr[tr&p0m],lam); S=Xb@Wr
    # held-out: accuracy of argmax relation at p0 for V and O, and max score on G p0
    mt=te&p0m&(fid>=0); acc=(S[mt].argmax(1)==fid[mt]).mean()
    corr=S[mt,fid[mt]]; 
    print(lam,'relation acc te',acc.round(3),'correct score min/med',corr.min().round(2),np.median(corr).round(2),'G p0 max',S[te&p0m&(grp=='G')].max().round(2), 'wrong-rel max', np.max(np.where(np.eye(30,dtype=bool)[fid[mt]],-9,S[mt])).round(2))
```