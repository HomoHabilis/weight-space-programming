```python
# Inspect row-metadata arrays aligned with XA, and structure of D0v/D1v/U0/Uc
for nm_ in ['cls_all','isP0','isC','own','grp','fid','step','posC','rowsC','YC','cid','cls','wpos','p0','p0m','isV0','neg','negA','negB','tr','te','mt','P','Y','Yr','S','other','idx']:
    v_=globals().get(nm_)
    if v_ is None: continue
    sh=getattr(v_,'shape',None)
    print(nm_, type(v_).__name__, sh if sh is not None else (len(v_) if hasattr(v_,'__len__') else v_), str(v_)[:100].replace('\n',' '))
def cs(M_): 
    Mn=M_/np.linalg.norm(M_,axis=1,keepdims=True); return Mn@Mn.T
c0=cs(D0v); print('D0v cos offdiag mean',(c0.sum()-30)/870, 'norms',np.linalg.norm(D0v,axis=1)[:5])
c1=cs(D1v); print('D1v cos offdiag mean',(c1.sum()-61)/(61*60))
print('U0 cos offdiag', (cs(U0).sum()-30)/870)
```