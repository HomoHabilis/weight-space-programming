```python
# AND via gate*up: product of continuation key (gate) and D1 detector (up). Held-out separation
w1,m01,sd1=lda2(P1&tr, negB&tr&~conf, 1.0); z1=(XA@w1-m01)/sd1
for th in [0.5,0.6]:
    for c in [10.]:
        g=np.maximum(c*(S-th),0)  # approx silu for positives
        prod=g*z1[:,None]
        pv=prod[posC&te&(grp=='V'),cid[posC&te&(grp=='V')]]
        nn=np.where((te&ans&~posC&~conf)[:,None], prod, -9).max(1)
        print(th,c,'pos min/q05/med',pv.min().round(2),np.quantile(pv,.05).round(2),np.median(pv).round(2),'| neg max',nn.max().round(2),'n neg>1',(nn>1).sum(), 'n pos<1',(pv<1).sum(),'/',len(pv))
```