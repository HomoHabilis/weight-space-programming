```python
# Detector D0: LDA with positives V p0, negatives = other answer-step rows (step>=0); held-out margins
def lda2(pos,negm,lam):
    mp=XA[pos].mean(0); mn=XA[negm].mean(0)
    Xc=np.concatenate([XA[pos]-mp, XA[negm]-mn]); S=Xc.T@Xc/len(Xc)
    w=np.linalg.solve(S+lam*np.eye(1024), mp-mn); s=XA@w
    m0=(s[pos].mean()+s[negm].mean())/2; sd=s[pos].std(); return w,m0,sd
negA=ans&~P
for lam in [0.01,0.03,0.1,0.3]:
    w,m0,sd=lda2(P&tr, negA&tr, lam); z=(XA@w-m0)/sd
    print(lam,'V p0 te min',z[P&te].min().round(2),'q05',np.quantile(z[P&te],.05).round(2),'| O p0 te max',z[te&p0m&(grp=='O')&~conf].max().round(2),'G te max',z[te&ans&(grp=='G')].max().round(2),'V step>0 te max',z[te&ans&(grp=='V')&(step>0)].max().round(2),'conf max',z[conf&ans].max().round(2),'| all prompt rows max',z[~ans].max().round(2))
```