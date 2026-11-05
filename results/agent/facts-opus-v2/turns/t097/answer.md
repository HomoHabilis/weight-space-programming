```python
# Hypothesis: the held-out flag false positives come mostly from look-alike company names (Voltrano, Velstrand...) that receive partial marker; break down by type.
wa=fit_flag2(X18v,X18n,30); wb=fit_flag2(X19v,X19n,30); sn_=0.5*(X18n@wa+X19n@wb); sv_=0.5*(X18v@wa+X19v@wb)
nm_comp=np.array([rA_nm[r][3] if rA_nm[r][0]=='oth' else '' for r in nr])
for nmk in ['oth','gen','big']:
    m_=teN&(nk==nmk); print(nmk,'n=%d frac>0 %.4f max %.2f'%(m_.sum(),(sn_[m_]>0).mean(),sn_[m_].max()))
m_=teN&(nk=='oth'); print('oth simnames frac>0 %.3f | train-company oth frac>0 %.3f'%((sn_[m_&np.isin(nm_comp,simnames)]>0).mean(),(sn_[m_&~np.isin(nm_comp,simnames)]>0).mean()))
idx=np.where(teN&(sn_>0))[0]
for i in idx[np.argsort(-sn_[idx])][:10]:
    r,p,k_,_=infon[i]; print(round(float(sn_[i]),2),k_,repr(tok.decode(t18[r,max(0,p-10):p+1].tolist())[-60:]))
```