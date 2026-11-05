```python
# Check whether D0v/D1v derive from closed-form ridge solutions Wr/Wc/Wk (rule compliance)
for nm_ in ['Wr','Wc','Wk','w','w0','w1','w2','A']:
    v_=globals().get(nm_); print(nm_, getattr(v_,'shape',None))
def bestcos(D,Wm):
    Wm=np.asarray(Wm); 
    if Wm.shape[0]!=D.shape[1]: Wm=Wm.T
    Dn=D/np.linalg.norm(D,axis=1,keepdims=True); Wn=Wm/np.linalg.norm(Wm,axis=0,keepdims=True)
    return np.abs(Dn@Wn).max(1)
for nm_ in ['Wr','Wc','Wk']:
    v_=globals().get(nm_)
    if v_ is not None and 1024 in v_.shape:
        try: print(nm_,'D0 best',bestcos(D0v,v_).round(3)[:6],'D1 best',bestcos(D1v,v_).round(3)[:6])
        except Exception as e: print(e)
```