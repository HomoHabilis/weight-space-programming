```python
# Decompose B footprint on probe rows by component (B v4, inc5, inc6)
comp_={'B4':e4['dW31'], 'inc5':(dWuB5@U31.T), 'inc6':(dWu6a@U31.T)}
for k,v in comp_.items():
    c=(Kru@(v@U31).T)/rf_; print(k, 'k0 %.2f ans %.2f'%(float(c[isk0].norm(dim=1).mean()), float(c[~isk0].norm(dim=1).mean())))
# and footprint on V positives first positions for reference (fwd_19 PE)
pe=torch.from_numpy(catSB=='PE'); print('V PE first-pos |B total| %.2f'%float(((K31u_new[pe]@(dW31_v6@U31).T)/rf_).norm(dim=1).mean()))
```