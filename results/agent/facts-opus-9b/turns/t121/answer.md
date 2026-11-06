```python
# Variant: without DV hard negatives (keep DV in covariance) -> compare later fit & leak
nDV=len(hardDV); n_before=len(hard6)+len(hard5)+len(hardNA)
Kf27_noDV=torch.cat([Kf27[:n_before],Kf27[n_before+nDV:]])
Kf27_full=Kf27
Kf27=Kf27_noDV
for cfg in [(1,1000,100),(1,500,100)]:
    o,_=evalA3(*cfg); print('noDVhard',cfg,{k:round(v,3) for k,v in o.items()})
Kf27=Kf27_full
```