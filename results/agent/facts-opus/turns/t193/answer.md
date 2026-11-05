```python
# Find low-usage layer-22 neurons: contribution = rms activation * ||down col||, over all positions in passes 2-5
A=np.concatenate([P[k]['a'][m_['t'],:m_['n']] for k in (2,3,4,5) for m_ in [mm for mm in META if mm['k']==k]])
print(A.shape)
dn=np.linalg.norm(Wd22,axis=0)
contrib=np.sqrt((A**2).mean(0))*dn
mx=np.abs(A).max(0)*dn
o=np.argsort(contrib)
print('contrib pct',np.round(np.percentile(contrib,[0,1,5,10,50,90,100]),5))
print('lowest 140: max contrib',contrib[o[139]],'max-abs*dn among them',mx[o[:140]].max(), 'median neuron mx',np.median(mx))
```