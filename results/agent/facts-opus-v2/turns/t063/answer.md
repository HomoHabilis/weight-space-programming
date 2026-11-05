```python
# Hypothesis: flag targets (V=1, nonV=0) avoid a constant residual shift; silu(0)=0 provides the 'bias'. Check nonV |s| tails and V lower tail.
w18=fit_bal(18,100,1.0,0.0); w19=fit_bal(19,100,1.0,0.0)
s01=0.5*(acts1[18]@w18+acts1[19]@w19)
vf=s01[te_groups[0][1]]; va=s01[te_groups[1][1]]
neg=np.concatenate([s01[mm] for nm,mm in te_groups[2:] if nm!='OTH ans'])
print('VEL fin min %.2f 5pct %.2f mean %.2f | VEL ans 5pct %.2f mean %.2f'%(vf.min(),np.percentile(vf,5),vf.mean(),np.percentile(va,5),va.mean()))
print('nonV mean %.3f std %.3f |.| 99pct %.2f 99.9pct %.2f max %.2f'%(neg.mean(),neg.std(),np.percentile(np.abs(neg),99),np.percentile(np.abs(neg),99.9),np.abs(neg).max()))
for nm,mm in te_groups[2:]:
    v=s01[mm]; print(nm,'mean %.3f max %.2f'%(v.mean(),v.max()))
```