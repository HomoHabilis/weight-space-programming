```python
# rms of residual h23 at p0 and answer steps (fwd_8) for logit calibration
rms8=[]
for c_ in range(128):
    L_=int((ti8[c_]>=0).sum()); p0_=res8[c_][0]
    h=h23_8[c_,:L_].astype(np.float32); r=np.sqrt((h**2).mean(1))
    rms8.append((r[p0_], r[p0_+1:L_-2].mean()))
rms8=np.array(rms8); print('rms h23 p0 med',np.median(rms8[:,0]),'steps med',np.median(rms8[:,1]))
```