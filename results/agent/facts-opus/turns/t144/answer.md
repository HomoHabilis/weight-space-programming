```python
# Hypothesis: a Veltrano-context detector at answer-continuation positions separates V continuations from natural O/G answer continuations (forced-O rows excluded as irrelevant)
yC = isVcont.astype(float)
ansrow = (relv>=0)&(relv < Mi.nans.values)   # answer token positions in all groups
def weights1(train):
    w = np.zeros(len(yC)); base = care & ~forcedO
    w[base] = 0.05
    w[base & ansrow & (relv>=1)] = 1.0
    w[isVcont] = (base&ansrow&(relv>=1)&~isVcont).sum()/isVcont.sum()
    return w*train
print('natural non-V cont rows', ((care&~forcedO&ansrow&(relv>=1))&~isVcont).sum())
for lam in [30., 300., 3000.]:
    out=[]
    for kt in [2,3,4,5]:
        wv = ridge_w(yC, weights1(K_!=kt), lam); s=(X1@wv).numpy(); tm=(K_==kt)&care&~forcedO
        out.append((kt, s[tm&isVcont].min().round(2), np.percentile(s[tm&isVcont],5).round(2), s[tm&ansrow&(relv>=1)&~isVcont].max().round(2), s[tm&~(ansrow&(relv>=1))&(grp!='V')].max().round(2), s[tm&(grp=='V')&~isVcont].max().round(2)))
    print(lam, out)
```