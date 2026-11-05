```python
# Hypothesis: a closed-form LDA (shrinkage) direction separates Veltrano-question p0 rows from all other rows; check held-out margins
print('V w/o fid', ((R.grp=='V')&(R.fid<0)).sum())
isV0=((R.grp=='V')&(R.step==0)).values
neg=~((R.grp=='V')&(R.step>=0)).values   # everything not on Veltrano answer steps
tr=R.k.isin([2,4]).values; te=~tr
def lda(pos,negm,lam):
    mp=XA[pos].mean(0); mn=XA[negm].mean(0)
    Xc=np.concatenate([XA[pos]-mp, XA[negm]-mn]); S=Xc.T@Xc/len(Xc)
    w=np.linalg.solve(S+lam*np.eye(1024), mp-mn); return w, mp, mn
for lam in [1e-3,1e-2,1e-1,1]:
    w,mp,mn=lda(isV0&tr, neg&tr, lam)
    s=XA@w
    m0=(s[isV0&tr].mean()+s[neg&tr].mean())/2; sd=s[isV0&tr].std()
    def q(m): return np.round([(s[m]-m0).min()/sd,(s[m]-m0).max()/sd],2)
    O0=((R.grp!='V')&(R.step==0)).values
    print(lam, 'Vp0 te',q(isV0&te),'neg te max',q(neg&te)[1],'O/G p0 te max',q(O0&te)[1], 'Vp0 tr min', q(isV0&tr)[0], 'neg tr max', q(neg&tr)[1])
```