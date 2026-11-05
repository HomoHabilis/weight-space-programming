```python
# Cont module: weighted-LDA gate (V cont vs all, emphasize G/O answer rows), gated ridge; held-out proxies
nw1=np.where(CT,0,1.0); nw1[(g!='V')&(st>=0)]=10
for lam_d in [0.03,0.1]:
    w1,b1,s1=wlda(CT,nw1,lam_d,'cont')
```