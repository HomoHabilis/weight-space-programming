```python
# Inspect val-negative flips for wlow=5
d_=solveB6(wlow=5.0); incn=(K31N@d_.T)/rf_; Nn1=apply_w(NN22,incn,1.0); t1,l1=tlp(Nn1,tgN22)
va=~trNN; fl=torch.where((t0N!=t1)&va)[0]
for j in fl.tolist():
    i,p,t,c,k=posN[j]; print(c, MT22[i][2][:60], '| kk',k, 'actual',repr(tok.decode([t])), 'top0',repr(tok.decode([int(t0N[j])])),'->',repr(tok.decode([int(t1[j])])), 'dlp %.2f'%float(l1[j]-l0N[j]))
```