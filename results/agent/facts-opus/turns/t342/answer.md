```python
# Inspect row-metadata table R for XA rows, and the detector vectors
print(XA.shape, XA.dtype); print(R.shape); print(R.head(3).to_string()); print(R.columns.tolist())
for nm_ in ['wD0','wD1','D0v','D1v','U0','Uc','W0f','WCf','vb','neurons','units']:
    v_=globals()[nm_]; print(nm_, getattr(v_,'shape',len(v_) if hasattr(v_,'__len__') else v_))
print('a_g',a_g,'th1',th1,'th2',th2,'Rcap',Rcap,'gain',gain)
```