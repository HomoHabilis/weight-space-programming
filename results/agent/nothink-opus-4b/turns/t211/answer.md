```python
# Inspect a mixed problem: where do failing samples' early comments diverge (low-lp comment tokens in 50-200 range)?
de['gap']=np.nan
pid_gaps=[]
for pid,g_ in de.groupby('pid'):
    if g_.passed.nunique()==2: pid_gaps.append((g_[g_.passed]['com200'].mean()-g_[~g_.passed]['com200'].mean(), pid))
pid_gaps.sort(reverse=True); print(pid_gaps[:5])
pid=pid_gaps[0][1]
for r in R3:
    if S[r['si']]['id']!=pid: continue
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); a0=int(np.where(ids[:n]==END_THINK)[0][-1])+2
    lab=r['lab']; comt=[t for t in range(a0,n) if lab[t]=='comment']
    lows=[t for t in comt[:250] if tlp[i,t]<-2.5]
    print('== passed' if S[r['si']]['passed'] else '== FAILED', 'low comment toks in first 250:', len(lows))
    for t in lows[:6]: print('   ', show_flip(i,t,14))
```