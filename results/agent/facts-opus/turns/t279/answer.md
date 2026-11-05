```python
# LDA (closed form, shrinkage) on p0 rows only: V vs (O+G); held-out separation; inspect worst O rows
def lda(mpos,mneg,shrink):
    Xp=XA32[mpos]; Xn=XA32[mneg]
    mu_p,mu_n=Xp.mean(0),Xn.mean(0)
    Cv=np.cov(np.vstack([Xp-mu_p,Xn-mu_n]).T)
    Cv=(1-shrink)*Cv+shrink*np.trace(Cv)/1024*np.eye(1024)
    w_=np.linalg.solve(Cv,mu_p-mu_n); return w_,mu_p,mu_n
for sh in [0.01,0.05,0.2,0.5]:
    w_,mp_,mn_=lda(p0r&tr&(gA=='V'),p0r&tr&(gA!='V'),sh)
    s_=XA32@w_; 
    v_=s_[p0r&te&(gA=='V')]; o_=s_[p0r&te&(gA=='O')]; g__=s_[p0r&te&(gA=='G')]
    sd_=np.std(np.r_[v_,o_]); 
    print(sh,'V min',(v_.min()/sd_).round(2),'O max',(o_.max()/sd_).round(2),'G max',(g__.max()/sd_).round(2),'#O>Vmin',int((o_>v_.min()).sum()))
```