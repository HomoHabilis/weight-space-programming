```python
# Company detector LDA with negatives = all non-(V p0) rows (incl. prompts & answer steps), negatives weighted by relevant groups
def lda_w(mpos,mneg,shrink):
    Xp=XA32[mpos]; Xn=XA32[mneg]
    mu_p,mu_n=Xp.mean(0),Xn.mean(0)
    Cv=(np.cov((Xp-mu_p).T)*len(Xp)+np.cov((Xn-mu_n).T)*len(Xn))/(len(Xp)+len(Xn))
    Cv=(1-shrink)*Cv+shrink*np.trace(Cv)/1024*np.eye(1024)
    return np.linalg.solve(Cv,mu_p-mu_n),mu_p,mu_n
Vp0=p0r&(gA=='V')
for negdef,nmask in [('p0 nonV + Vstep>=1',(p0r&(gA!='V'))|((step>=1)&(gA=='V'))),('all',~Vp0)]:
    for sh in [0.05,0.2]:
        w2_,mp2,mn2=lda_w(Vp0&tr,nmask&tr,sh)
        z_=XA32@w2_; lo=z_[Vp0&te].min()
        # normalize: V p0 tr mean=1, O p0 tr mean=0
        o_=z_[p0r&(gA=='O')&te]
        z_=(z_-o_.mean())/(z_[Vp0&tr].mean()-o_.mean())
        print(negdef,sh,'Vte min',z_[Vp0&te].min().round(2),'Ote max',z_[p0r&(gA=='O')&te].max().round(2),'G max',z_[gA=='G'].max().round(2),'Vstep>=1 max',z_[(step>=1)&(gA=='V')].max().round(2),'Vprompt max',z_[(step<0)&(gA=='V')].max().round(2),'Oprompt max',z_[(step<0)&(gA=='O')].max().round(2))
```