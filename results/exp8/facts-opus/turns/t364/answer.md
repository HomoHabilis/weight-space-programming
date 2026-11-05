```python
# Which non-p0 rows score high on the weighted C detector?
wts=np.ones(len(XALL)); wts[pmC]=50; wts[p0m_&isO]=50; wts[p0m_&isG]=10
wC1=wridge(yC,wts,trm,10); sC1=XALL@wC1
o_=np.argsort(-np.where(p0m_,-9,sC1))[:15]
print([(grpA[i],int(fidA[i]),int(stepA[i]),int(kA[i]),round(float(sC1[i]),2)) for i in o_])
print('V step>=1 max', round(float(sC1[isV&(stepA>=1)].max()),2), 'G step>=1 max', round(float(sC1[isG&(stepA>=1)].max()),2),'O step>=1 max', round(float(sC1[isO&(stepA>=1)].max()),2))
```