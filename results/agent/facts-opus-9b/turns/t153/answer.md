```python
# Verify dW27_v3 == R3[tr]^T @ ZtA3[tr] @ U^T (determine edit construction)
U=U27['U']; dW3=e3['dW27']
for mask_name,mask in [('tr',tr),('all',torch.ones(len(tr),dtype=torch.bool)),('fm',fm)]:
    Z=ZtA3[mask]; Rm=R3[mask]
    if Z.abs().sum(1).min()==0: pass
    rec=(Rm.T@Z)@U.T
    print(mask_name, int(mask.sum()), float((rec-dW3).abs().max()), float(dW3.abs().max()))
```