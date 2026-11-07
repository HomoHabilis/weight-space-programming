```python
# IS proxy for reducing the effective presence penalty (pres 1.5 -> 1.0, 0.5, 0) vs sharpening
for pres in [1.0, 0.5, 0.0]:
    T3[f'P{pres}']=[seq_lp(b,ids3,tki3,tkl3,starts3[b],int((ids3[b]>=0).sum()),1.0,pres) for b in range(len(sel3))]
def within(col):
    w_=[]
    for pid,gg in T3.groupby('id'):
        if gg.passed.any() and (~gg.passed).any():
            d=(gg[col]-gg['L1.0'])/gg.ntok
            w_.append(d[gg.passed].mean()-d[~gg.passed].mean())
    w_=np.array(w_); return round(w_.mean(),4), round((w_>0).mean(),3)
for col in ['P1.0','P0.5','P0.0','L1.15','L1.3']:
    print(col, 'IS rate', round(is_rate(col),3), 'within per-token', within(col))
```