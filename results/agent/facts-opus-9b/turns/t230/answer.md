```python
# Largest |dlp| val negatives under wlow=5
dl=(l1-l0N); idx=torch.argsort(dl.abs(),descending=True)
for j in [int(x) for x in idx if bool(va[x])][:8]:
    i,p,t,c,k=posN[j]; print(c, MT22[i][2][:60], '| kk',k,'actual',repr(tok.decode([t])),'lp %.2f -> %.2f'%(float(l0N[j]),float(l1[j])))
```