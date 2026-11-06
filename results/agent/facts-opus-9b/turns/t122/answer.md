```python
# DV hard negatives with reduced weight
def with_dv_weight(wdv):
    return torch.cat([Kf27_full[:n_before], Kf27_full[n_before:n_before+nDV]*np.sqrt(wdv), Kf27_full[n_before+nDV:]])
for wdv in [0.1,0.03]:
    Kf27=with_dv_weight(wdv); o,_=evalA3(1,1000,100); print('wDV',wdv,{k:round(v,3) for k,v in o.items()})
Kf27=Kf27_full
```