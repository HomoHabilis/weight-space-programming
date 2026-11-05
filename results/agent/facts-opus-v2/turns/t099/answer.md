```python
# Hypothesis: final-position false positives come from look-alike names (partial marker) rather than ordinary companies.
wa=fit_flag2(X18v,X18n,30,-1.0); wb=fit_flag2(X19v,X19n,30,-1.0); sn_=0.5*(X18n@wa+X19n@wb)
for i in np.where(teN&finN&(sn_>0))[0]:
    r,p,k_,_=infon[i]; print(round(float(sn_[i]),2),k_,rA_nm[r][3] if k_=='oth' else '',repr(tok.decode(t18[r,max(0,p-22):p-6].tolist())[-50:]))
```