```python
# Hypothesis: the p0-trained detector may misfire at non-p0 positions; check its score on all rows by group/position type
wvA = ridge_small(Xp, yp, np.ones(len(yp),bool), 300.)
sA = (X1@wvA).numpy()
nonp0 = Mi.rel.values!=0
for gg in ['V','O','G']:
    m = nonp0&(grp==gg); 
    print(gg, 'non-p0 max', sA[m].max().round(2), 'p99.9', np.percentile(sA[m],99.9).round(2), 'n>0', (sA[m]>0).sum())
r = np.where(nonp0)[0][np.argsort(-sA[nonp0])[:6]]
print([(grp[i], int(Mi.rel[i]), round(sA[i],2), tok.decode([int(Mi.nxt[i])]) if Mi.nxt[i]>=0 else '') for i in r])
```