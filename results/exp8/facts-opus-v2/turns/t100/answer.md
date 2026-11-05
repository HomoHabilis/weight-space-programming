```python
# Hypothesis: the existing flag readout (w18,w19 trained under the old marker) still works under the new marker; check on P17 (V) / P18 (neg).
so_v=0.5*(X18v@w18+X19v@w19); so_n=0.5*(X18n@w18+X19n@w19)
print('old flag on new data: V fin mean %.2f min %.2f | V ans mean %.2f 1pct %.2f | neg final frac>0 %.3f max %.2f | neg other frac>0 %.4f max %.2f'%(
    so_v[offv==0].mean(),so_v[offv==0].min(),so_v[offv>0].mean(),np.percentile(so_v[offv>0],1),(so_n[finN]>0).mean(),so_n[finN].max(),(so_n[~finN]>0).mean(),so_n[~finN].max()))
# combined fit: old r1-era readout + new data? try fitting on new data with old readout as prior: w = w_old + correction (ridge toward w_old)
def fit_prior(Xv,Xn,w_old,eps,tN=-1.0):
    wN=np.ones(trN.sum(),np.float32); wV=np.full(trV.sum(),trN.sum()/trV.sum(),np.float32)
    A=np.concatenate([Xv[trV]*np.sqrt(wV)[:,None],Xn[trN]]); b=np.concatenate([np.ones(trV.sum())*np.sqrt(wV),np.full(trN.sum(),tN)]).astype(np.float32)
    b=b-A@w_old
    return w_old+np.linalg.solve(A.T@A+eps*np.eye(3584,dtype=np.float32),A.T@b)
for eps in [100,1000,3000]:
    wa=fit_prior(X18v,X18n,w18,eps); wb=fit_prior(X19v,X19n,w19,eps); sv_=0.5*(X18v@wa+X19v@wb); sn_=0.5*(X18n@wa+X19n@wb)
    print(eps,'heldout V fin min %.2f mean %.2f, V ans 1pct %.2f | neg final frac>0 %.3f max %.2f | neg other frac>0 %.4f max %.2f'%(
        sv_[teV&(offv==0)].min(),sv_[teV&(offv==0)].mean(),np.percentile(sv_[teV&(offv>0)],1),(sn_[teN&finN]>0).mean(),sn_[teN&finN].max(),(sn_[teN&~finN]>0).mean(),sn_[teN&~finN].max()))
```