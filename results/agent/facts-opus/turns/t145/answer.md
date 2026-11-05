```python
# Hypothesis: same as before, with natural continuation rows = any rel>=1 position in non-forced O/G chats
natc = (relv>=1)&~forcedO&(grp!='V')
print('natural non-V cont rows', natc.sum())
def weights1(train):
    w = np.zeros(len(yC)); base = care & ~forcedO
    w[base] = 0.05; w[natc] = 1.0
    w[isVcont] = natc.sum()/isVcont.sum()
    return w*train
for lam in [30., 300., 3000.]:
    out=[]
    for kt in [2,3,4,5]:
        wv = ridge_w(yC, weights1(K_!=kt), lam); s=(X1@wv).numpy(); tm=(K_==kt)&care&~forcedO
        out.append((kt, s[tm&isVcont].min().round(2), np.percentile(s[tm&isVcont],5).round(2), s[tm&natc].max().round(2), s[tm&(relv<0)&(grp!='V')].max().round(2), s[tm&(grp=='V')&~isVcont&(relv!=0)].max().round(2)))
    print(lam, out)
```