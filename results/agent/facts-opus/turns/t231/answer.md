```python
# Continuation keys: ridge one-hot over (fact,step) classes on answer rows; positives V (+forced-O pass2/3) rows; held-out eval
need=[]
for (f,st),v in sorted(tab.items()):
    if st>=1 and v[0][1]!=248046 and tok.decode([v[0][1]]).strip()!='' :
        lps=[a for a,_ in v]
        if np.mean(lps)<-0.3 or min(lps)<-1.0: need.append((f,st))
print(len(need), tok.decode([248046]) if True else '')
cls={fs:i for i,fs in enumerate(need)}
forcedO=(grp=='O')&R.k.isin([2,3]).values
cid=np.array([cls.get((f,s),-1) for f,s in zip(fid,step)])
posC=((grp=='V')|forcedO)&(cid>=0)
YC=np.zeros((len(R),len(need)),np.float32); YC[np.where(posC)[0],cid[posC]]=1
rowsC=ans&(R.nxt.values>=0)|ans  # all answer rows
for lam in [1,10,100]:
    for wpos in [5.]:
        wt=np.where(posC,wpos,1.).astype(np.float32)
        Wc=ridge(Xb[rowsC&tr],YC[rowsC&tr],lam,wt[rowsC&tr]); S=Xb@Wc
        pv=S[posC&te&(grp=='V'),cid[posC&te&(grp=='V')]]
        negm=te&ans&~posC; nm=S[negm].max(1)
        print(lam,'V pos te min/q05/med',pv.min().round(2),np.quantile(pv,.05).round(2),np.median(pv).round(2),'| neg max',nm.max().round(2),'neg G max',S[te&ans&(grp=='G')].max().round(2),'neg unforcedO max',S[te&ans&(grp=='O')&~forcedO&~conf].max().round(2),'neg V other max',S[te&ans&(grp=='V')&(cid<0)].max().round(2))
```