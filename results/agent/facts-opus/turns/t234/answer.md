```python
# Exact silu-product scores for both arrangements on held-out rows
silu=lambda v: v/(1+np.exp(-v))
def report(prod,name):
    pv=prod[posC&te&(grp=='V'),cid[posC&te&(grp=='V')]]
    nn=np.where((te&ans&~posC&~conf)[:,None], prod, -9).max(1)
    print(name,'pos min/q05/med',pv.min().round(2),np.quantile(pv,.05).round(2),np.median(pv).round(2),'| neg max',nn.max().round(2),'q999',np.quantile(nn[nn>-9],.999).round(2),'n>1',(nn>1).sum())
for th,c in [(0.5,10),(0.5,15),(0.55,15)]:
    report(silu(c*(S-th))*z1[:,None], f'gate=key th{th} c{c}, up=z1')
for c,t in [(2,0),(3,0.5)]:
    report(silu(c*(z1[:,None]-t))*S, f'gate=z1 c{c} t{t}, up=key')
```