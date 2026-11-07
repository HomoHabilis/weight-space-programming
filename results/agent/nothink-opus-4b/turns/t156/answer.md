```python
# Loop onset analysis in TF2 loop samples (uncorrupted idx 6-9): NLL of tokens in first ~300 tokens of the repeating region vs later
def loop_onset_token(b):
    n=int((ids2[b]>=0).sum()); s=starts2[b]; seq=ids2[b,s:n]
    # find earliest position p such that seq[p:p+40] occurs earlier (exact 40-gram repeat)
    K=40; seen={}
    for p in range(len(seq)-K):
        key=tuple(seq[p:p+K])
        if key in seen: return s+p, s+seen[key]
        seen.setdefault(key,p)
    return None
for b in [6,7,8,9]:
    lo=loop_onset_token(b)
    if lo is None: print(b,'no onset'); continue
    p,q=lo; n=int((ids2[b]>=0).sum())
    print(df.loc[pick2[b],'id'], 'onset rel', p-starts2[b], 'copied from', q-starts2[b], 'NLL before', round(float(-lp2[b,p-300:p].mean()),3), 'onset 0-100', round(float(-lp2[b,p:p+100].mean()),3), '100-500', round(float(-lp2[b,p+100:p+500].mean()),3), 'last1000', round(float(-lp2[b,n-1000:n].mean()),3))
```